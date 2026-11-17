"""Two-direction source-only training, calibration and late-fusion benchmark."""
from .external_paths import allowed_output
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from .data import load_dfl, sha256
from .models import features, baseline, rules, RuleConfig
from .evaluation import official, evaluate_period, aggregate
from .run import save_json, save_table
from .stress import perturb
from .probabilistic import (CONFIGS, STRIDE, NAMES, build, geometry, make_sequence, split_ranges, expand,
    temperature, fit_temperature, probability_metrics, fit_weights, transition_parameters, infer_chain, normalize)


def score(pm, sequences, prediction):
    return aggregate(pm, [evaluate_period(pm, s.full_y, y) for s, y in zip(sequences, prediction)])


def predict(model, sequences):
    return [model.predict(s) for s in sequences]


def concat(values):
    return np.concatenate(values)


def fusion(probabilities, weights, transition, prior, strength):
    p = normalize(sum(w*x for w, x in zip(weights, probabilities)))
    if strength == 0:
        return p, p.argmax(axis=1).astype(np.int8)
    trans = transition**strength
    trans /= trans.sum(axis=1, keepdims=True)
    return infer_chain(np.log(np.maximum(p, 1e-9))-np.log(prior), trans, prior)


def directions(examples, train_id, test_id, out, pm):
    folder = out/(train_id+"_to_"+test_id)
    folder.mkdir()
    models_dir = folder/"models"
    models_dir.mkdir()
    train, test = examples[train_id], examples[test_id]
    parts = {key: [] for key in ("fit", "select", "refit", "calibrate")}
    partition_record = []
    for data, y, x in train:
        ranges = split_ranges(len(y))
        partition_record.append({"period": data.period, "ranges": ranges, "frame_count": len(y)})
        for key, (a, b) in ranges.items():
            parts[key].append(make_sequence(data.period, x, y, a, b))
    test_sequences = [make_sequence(d.period, x, y) for d, y, x in test]
    save_json(folder/"protocol.json", {"train_match": train_id, "test_match": test_id,
        "partitions": partition_record, "purge_frames": 125, "stride": STRIDE, "configs": CONFIGS,
        "selection_metric": "source select-block pooled spellF1", "calibration": "source final 20% per half",
        "refit": "source first 80%, separated by 5s purge from calibration"})
    validation = []
    for name, (family, kwargs) in CONFIGS.items():
        started = time.perf_counter()
        model = build(name).fit(parts["fit"], models_dir/(name+"_selection"))
        result = predict(model, parts["select"])
        s = score(pm, parts["select"], [expand(y, seq) for seq, (_, y) in zip(parts["select"], result)])
        validation.append({"model": name, "family": family, "spell_f1": s["spell_f1"],
                           "macro_f1": s["macro_f1"], "seconds": time.perf_counter()-started})
        print(f"{train_id} selection {name}: spellF1={s['spell_f1']:.4f}", flush=True)
        del model
    chosen = {family: max([r for r in validation if r["family"] == family], key=lambda r: r["spell_f1"])["model"]
              for family in ("fuzzy", "hmm", "crf")}
    save_json(folder/"selection_before_test.json", {"validation": validation, "chosen": chosen})
    print(f"{train_id} selected from SOURCE ONLY: {chosen}", flush=True)
    rows, selected_models, calibration_outputs, test_outputs = [], {}, {}, {}
    confidence_records = {}
    artifact = {}
    cal_y = concat([s.y for s in parts["calibrate"]])
    test_y = concat([s.full_y for s in test_sequences])
    for name, (family, kwargs) in CONFIGS.items():
        model = build(name).fit(parts["refit"], models_dir/(name+"_final"))
        cal = predict(model, parts["calibrate"])
        test_pred = predict(model, test_sequences)
        temp = fit_temperature(concat([p for p, _ in cal]), cal_y)
        full_pred = [expand(y, s) for s, (_, y) in zip(test_sequences, test_pred)]
        full_raw = [expand(p, s) for s, (p, _) in zip(test_sequences, test_pred)]
        full_p = [temperature(p, temp) for p in full_raw]
        metric = score(pm, test_sequences, full_pred)
        before = probability_metrics(concat(full_raw), test_y, concat(full_pred))
        after = probability_metrics(concat(full_p), test_y, concat(full_pred))
        metric.update(model=name, family=family, temperature=temp, nll_raw=before["nll"], nll=after["nll"],
                      brier_raw=before["brier"], brier=after["brier"], ece_raw=before["ece"], ece=after["ece"],
                      source_selected=chosen[family] == name, role="single_model", selection_source="source_inner_validation")
        rows.append(metric)
        confidence_records[name] = {"raw": before, "calibrated": after}
        for s, p, y in zip(test_sequences, full_p, full_pred):
            artifact[name+"__"+s.key+"__p"] = p.astype(np.float32)
            artifact[name+"__"+s.key+"__label"] = y
        if chosen[family] == name:
            selected_models[family] = (model, temp)
            calibration_outputs[family] = [temperature(p, temp) for p, _ in cal]
            test_outputs[family] = [temperature(p, temp) for p, _ in test_pred]
        else:
            del model
        print(f"{train_id}->{test_id} {name}: spell={metric['spell_f1']:.4f} NLL={before['nll']:.3f}->{after['nll']:.3f}", flush=True)
    order = ("fuzzy", "hmm", "crf")
    learned = fit_weights([concat(calibration_outputs[k]) for k in order], cal_y)
    transition, prior = transition_parameters(parts["refit"])
    fusion_specs = {
        "ensemble_FH": ([.5, .5, 0], 0),
        "ensemble_HC": ([0, .5, .5], 0),
        "ensemble_uniform": ([1/3]*3, 0),
        "ensemble_weights": (learned.tolist(), 0),
        "ensemble_chain025": (learned.tolist(), .25),
        "ensemble_chain1": (learned.tolist(), 1.),
    }
    fusion_records = {}
    for name, (weights, strength) in fusion_specs.items():
        cal = [fusion([calibration_outputs[k][i] for k in order], weights, transition, prior, strength)
               for i in range(len(parts["calibrate"]))]
        source_metric = score(pm, parts["calibrate"],
                              [expand(y, s) for s, (_, y) in zip(parts["calibrate"], cal)])
        temp = fit_temperature(concat([p for p, _ in cal]), cal_y)
        test_pred = [fusion([test_outputs[k][i] for k in order], weights, transition, prior, strength)
                     for i in range(len(test_sequences))]
        full_pred = [expand(y, s) for s, (_, y) in zip(test_sequences, test_pred)]
        full_raw = [expand(p, s) for s, (p, _) in zip(test_sequences, test_pred)]
        full_p = [temperature(p, temp) for p in full_raw]
        result = score(pm, test_sequences, full_pred)
        before = probability_metrics(concat(full_raw), test_y, concat(full_pred))
        after = probability_metrics(concat(full_p), test_y, concat(full_pred))
        result.update(model=name, family="ensemble", temperature=temp, nll_raw=before["nll"], nll=after["nll"],
                      brier_raw=before["brier"], brier=after["brier"], ece_raw=before["ece"], ece=after["ece"],
                      role="ensemble", source_selected=False, selection_source="source_calibration_fusion_block")
        rows.append(result)
        fusion_records[name] = {"weights": weights, "chain_strength": strength,
                                "temperature": temp, "source_spell_f1": source_metric["spell_f1"]}
        confidence_records[name] = {"raw": before, "calibrated": after}
        for s, p, y in zip(test_sequences, full_p, full_pred):
            artifact[name+"__"+s.key+"__p"] = p.astype(np.float32)
            artifact[name+"__"+s.key+"__label"] = y
        print(f"{train_id}->{test_id} {name}: spell={result['spell_f1']:.4f}", flush=True)
    selected_ensemble = max(fusion_records, key=lambda name: fusion_records[name]["source_spell_f1"])
    for row in rows:
        if row["family"] == "ensemble":
            row["source_selected"] = row["model"] == selected_ensemble
    print("Ensemble selected on SOURCE calibration: "+selected_ensemble, flush=True)
    for name in ("B50", "rules_V0"):
        pred = [baseline(features(d), 50) if name == "B50" else
                rules(features(d), RuleConfig(.2, .08, .4, True)) for d, y, x in test]
        result = score(pm, test_sequences, pred)
        result.update(model=name, family="reference", role="reference", source_selected=False)
        rows.append(result)
    fields = ["model", "family", "role", "source_selected", "spell_f1", "macro_f1", "f1_D", "pred_spells",
              "gt_spells", "temperature", "nll_raw", "nll", "brier_raw", "brier", "ece_raw", "ece"]
    save_table(folder/"results.csv", rows, fields)
    save_json(folder/"results.json", {"train_match": train_id, "test_match": test_id, "rows": rows,
        "chosen_models": chosen, "selected_ensemble": selected_ensemble, "fusion": fusion_records,
        "confidence": confidence_records, "validation": validation})
    np.savez_compressed(folder/"probabilities.npz", **artifact)
    for (data, y, x), seq in zip(test, test_sequences):
        p = artifact[selected_ensemble+"__"+seq.key+"__p"]
        pred = artifact[selected_ensemble+"__"+seq.key+"__label"]
        save_table(folder/(seq.key+"_selected_confidence.csv"),
            ({"frame": i, "native_frame": int(data.frame_ids[i]), "label": "HAD"[int(pred[i])],
              "p_H": float(p[i, 0]), "p_A": float(p[i, 1]), "p_D": float(p[i, 2]),
              "confidence_of_label": float(p[i, pred[i]]),
              "entropy": float(-(p[i]*np.log(np.maximum(p[i], 1e-9))).sum())} for i in range(len(y))),
            ["frame", "native_frame", "label", "p_H", "p_A", "p_D", "confidence_of_label", "entropy"])
    stress_rows = []
    selected_spec = fusion_records[selected_ensemble]
    for seed in (0, 1, 2):
        seqs, altered = [], []
        for data, y, x in test:
            changed, _ = perturb(data, "missing_0.4", seed)
            altered.append(changed)
            seqs.append(make_sequence(changed.period, geometry(changed), y))
        member = {k: [temperature(p, selected_models[k][1])
                      for p, _ in predict(selected_models[k][0], seqs)] for k in order}
        mixed = [fusion([member[k][i] for k in order], selected_spec["weights"], transition, prior,
                         selected_spec["chain_strength"]) for i in range(len(seqs))]
        preds = [expand(y, s) for s, (_, y) in zip(seqs, mixed)]
        for name, output in [
            (selected_ensemble, preds),
            (chosen["crf"], [expand(y, s) for s, (_, y) in zip(seqs, predict(selected_models["crf"][0], seqs))]),
            ("rules_V0", [rules(features(d), RuleConfig(.2, .08, .4, True)) for d in altered]),
            ("B50", [baseline(features(d), 50) for d in altered])]:
            result = score(pm, seqs, output)
            result.update(model=name, seed=seed, scenario="missing_0.4")
            stress_rows.append(result)
        print(f"{train_id}->{test_id} stress seed {seed} complete", flush=True)
    save_table(folder/"stress.csv", stress_rows, ["model", "seed", "scenario", "spell_f1", "macro_f1", "f1_D"])
    save_json(folder/"fusion_artifact.json", {"chosen_models": chosen,
        "model_temperatures": {k: selected_models[k][1] for k in order},
        "selected_ensemble": selected_ensemble, "spec": selected_spec,
        "transition": transition.tolist(), "prior": prior.tolist(), "stride": STRIDE, "features": NAMES})
    plots(folder, rows, confidence_records, chosen, selected_ensemble)
    return {"train": train_id, "test": test_id, "rows": rows, "selected": chosen,
            "selected_ensemble": selected_ensemble, "fusion": fusion_records, "stress": stress_rows}


def plots(folder, rows, confidence, selected, ensemble):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in list(selected.values())+[ensemble]:
        data = confidence[name]["calibrated"]
        bins = [b for b in data["bins"] if b["count"] > 0]
        axes[0].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", label=name)
        risk = data["risk_coverage"]
        axes[1].plot([r["coverage"] for r in risk], [r["error_rate"] for r in risk], "o-", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(xlabel="Confidence of predicted label", ylabel="Observed accuracy", title="Test-match reliability", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Retained coverage", ylabel="Frame error rate", title="Confidence-based selective prediction")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder/"confidence.png", dpi=160)
    fig.savefig(folder/"confidence.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfl-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    if out.exists() or (not allowed_output(out, repo)):
        parser.error("--out must be new and under data/external/runs or outside the repository")
    out.mkdir(parents=True)
    pm = official(repo)
    examples, sources, audits = {}, [], {}
    for match in ("J03WMX", "J03WN1"):
        root = args.dfl_root/match
        metas, positions = list(root.glob("*matchinformation*"+match+".xml")), list(root.glob("*positions_raw_observed*"+match+".xml"))
        if len(metas) != 1 or len(positions) != 1:
            raise ValueError("Missing public DFL files for "+match)
        print("Loading public DFL "+match, flush=True)
        halves, audit = load_dfl(metas[0], positions[0])
        if audit["match_id"] != "DFL-MAT-"+match:
            raise ValueError("Match mismatch")
        audits[match] = audit
        examples[match] = [(data, y, geometry(data)) for data, y in halves]
        sources.extend({"path": str(p), "sha256": sha256(p)} for p in [metas[0], positions[0]])
    save_json(out/"audits.json", audits)
    save_json(out/"experiment_protocol.json", {
        "configs": CONFIGS, "stride": STRIDE, "seed": 0, "train_select_calibrate": "60/20/20, chronological within each half",
        "purge_seconds": 5, "refit": "first 80% of SOURCE half, before calibration purge",
        "calibration": "temperature by NLL on source final 20%; fusion weights and variant on same source block",
        "outer_evaluation": "both directions, no target labels used in fitting or selection",
        "limitations": ["Both games were previously examined during rule development; exploratory validation, not untouched final test.",
                        "Only two matches; confidence calibration need not transfer to Metrica ATD.",
                        "Source calibration block is reused for fusion fitting; outer-test evaluation is separate.",
                        "Offline prediction uses future context; not a live causal system.",
                        "No image/audio encoder and no competition data accessed."]})
    summaries = [directions(examples, a, b, out, pm) for a, b in (("J03WMX", "J03WN1"), ("J03WN1", "J03WMX"))]
    save_json(out/"summary.json", summaries)
    manifest = {"utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "python": __import__("sys").version, "command": __import__("sys").argv,
        "code_sha256": {str(p.relative_to(repo)): sha256(p) for p in sorted((repo/"possession_lab").glob("*.py"))},
        "scorer_sha256": sha256(repo/"evaluation"/"possession_metrics.py"),
        "dependencies": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "python-crfsuite", "matplotlib"]}}
    save_json(out/"manifest.json", manifest)
    lines = ["# Fuzzy, HMM, CRF i zespoły — dwa kierunki", "",
             "Wszystkie konfiguracje zapisane przed pomiarem. Wybór modelu, kalibracja i fuzja tylko na meczu źródłowym.",
             "Oznaczenie selected oznacza wybór na źródle, nigdy najlepszy wynik na meczu testowym.", ""]
    for summary in summaries:
        lines += [f"## {summary['train']} → {summary['test']}", "",
                  "| Model | selected | spellF1 | macroF1 | NLL przed/po | ECE przed/po |",
                  "|---|---|---:|---:|---|---|"]
        for r in summary["rows"]:
            nll = f"{r['nll_raw']:.3f}/{r['nll']:.3f}" if "nll" in r else "—"
            ece = f"{r['ece_raw']:.3f}/{r['ece']:.3f}" if "ece" in r else "—"
            lines.append(f"| {r['model']} | {r.get('source_selected', False)} | {r['spell_f1']:.4f} | {r['macro_f1']:.4f} | {nll} | {ece} |")
        lines += ["", f"Zespół wybrany na źródle: **{summary['selected_ensemble']}**.", ""]
    lines += ["## Interpretacja", "",
              "HMM jest nadzorowanym modelem Gaussa z trzema stanami H/A/D; parametry emisji i przejść estymujemy z etykiet źródłowych, bez EM.",
              "CRF to liniowy model łańcuchowy uczony L-BFGS, bez enkodera. Wariant bare używa bieżącej geometrii; kontekstowy także +/-0.4 s.",
              "Fuzzy używa miękkich funkcji przynależności i opcjonalnej pamięci; przynależność nie jest z definicji skalibrowanym prawdopodobieństwem.",
              "Pewność oznacza marginalne prawdopodobieństwo przewidzianej etykiety, nie prawdopodobieństwo poprawnej granicy posiadania.",
              "Temperatura zmienia pewność, ale nie zmienia etykiet pojedynczego modelu; może zmienić wynik późniejszej fuzji.",
              "Wnioskowanie co 2 klatki (12.5 Hz), ocena po odtworzeniu etykiet na oryginalnym 25 Hz; V0/B50 pozostają przy 25 Hz.",
              "Dwa uprzednio analizowane mecze DFL nie są niezależnym finalnym testem. Brak dowodu transferu na konkurs.",
              "DFL nie ma wideo: trening obrazu wymaga osobnego zbioru z prawami do nagrań i synchronizacją.",
              "Nie trenowano obrazu, nie otwierano danych konkursowych ani odłożonych meczów testowych."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("COMPLETE "+str(out/"REPORT.md"), flush=True)


if __name__ == "__main__":
    main()
