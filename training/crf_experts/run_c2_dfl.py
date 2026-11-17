"""Owner-run C2 A/B ablation. Import/--help never reads match inputs or trains.

Use --c1-run for the completed run_c1_local.py output (containing manifest.json
and experiment/). Models and results stay in NEW private runs outside the repository.
"""
from pathlib import Path
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import sys
import time
import uuid
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIVATE_RUNS = Path.home() / "Documents/tracking-physics-private/runs"
sys.dont_write_bytecode = True
sys.path[:0] = [str(HERE), str(ROOT)]

import numpy as np
from c2_dfl_features import make_sequence, NAMES
from possession_lab.data import sha256
from possession_lab.evaluation import official, evaluate_period, aggregate
from possession_lab.multimatch_run import MATCHES, partitions, load_completed
from possession_lab.probabilistic import STRIDE, temperature, fit_temperature
from possession_lab.probabilistic_run import fusion
from possession_lab.robust import RobustCRF, mask_ball
from possession_lab.robust_run import read_match
from possession_lab.stress import perturb
from possession_lab.strong_motion_noise import train_views

HELD = "J03WMX"
SOURCES = list(MATCHES[1:])
SCENARIOS = ("clean", "missing40", "noise_0.3m", "noise_plus_missing40")
METRICS = ("spell_f1", "macro_f1", "f1_D")
HISTORICAL_SPELL = dict(zip(SCENARIOS, (0.4615, 0.4348, 0.4643, 0.4162)))
ARTIFACTS = ("bundle.json", "model.crfsuite", "scaler.npz")


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)+"\n",
                    encoding="utf-8", newline="\n")


def safe_output(path):
    out = Path(path).resolve()
    private_runs = PRIVATE_RUNS.resolve()
    if (out == private_runs or not out.is_relative_to(private_runs)
            or out.is_relative_to(ROOT.resolve())):
        raise ValueError("Output must be a NEW subdirectory of tracking-physics-private/runs outside the repository")
    if out.exists():
        raise ValueError("Refusing to reuse/overwrite an existing output directory")
    return out


def validate_c1_manifest(manifest):
    if (manifest.get("status") != "completed"
            or manifest.get("matches_at_4_decimals") != 16
            or manifest.get("scores") != 16
            or manifest.get("held") != HELD
            or manifest.get("training_matches") != SOURCES
            or manifest.get("branch") != "codex/c1-sandbox"):
        raise ValueError("Expected the completed, matching C1 J03WMX reproduction manifest")


def validate_reference_scores(rows):
    expected = {(variant, scenario) for variant in ("baseline", "noise", "motion", "motion_noise")
                for scenario in SCENARIOS}
    keys = [(row["variant"], row["scenario"]) for row in rows]
    if len(keys) != 16 or set(keys) != expected:
        raise ValueError("C1 scores must contain exactly the 16 original cells")
    selected = {row["scenario"]: row for row in rows if row["variant"] == "motion_noise"}
    for scenario, row in selected.items():
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid C1 score: "+metric)
        if f"{row['spell_f1']:.4f}" != f"{HISTORICAL_SPELL[scenario]:.4f}":
            raise ValueError("C1 motion_noise differs from the historical table")
    return selected


def validate_baseline_config(config):
    required = {"training_matches": SOURCES, "stride": 2, "history": False,
                "augmentation": True, "variant": "augmentation",
                "c2": 0.1, "context": 5, "chain_strength": 0.25}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Baseline does not match the frozen C1 protocol")
    transition, prior = np.asarray(config["transition"]), np.asarray(config["prior"])
    if (transition.shape != (3, 3) or prior.shape != (3,)
            or not np.isfinite(transition).all() or not np.isfinite(prior).all()
            or np.any(transition <= 0) or np.any(prior <= 0)
            or not np.allclose(transition.sum(1), 1) or not np.isclose(prior.sum(), 1)):
        raise ValueError("Invalid frozen transition/prior")


def validate_partitions(records, config):
    if records != config["partitions"]:
        raise ValueError("Source partitions differ from C1; refusing to train")


def normalized_path(path):
    return str(Path(path).resolve()).casefold()


def c1_inputs(c1_run):
    """Called only by the explicit owner CLI, never during import/tests."""
    manifest_path = c1_run/"manifest.json"
    manifest = json_read(manifest_path)
    validate_c1_manifest(manifest)
    references = validate_reference_scores(json_read(c1_run/"experiment"/"scores.json"))
    command = manifest["command"]
    def argument(flag):
        if command.count(flag) != 1:
            raise ValueError("Missing/ambiguous flag in the C1 command: "+flag)
        value = Path(command[command.index(flag)+1])
        if not value.is_absolute():
            raise ValueError("Expected absolute C1 input paths")
        return value
    data, baseline = argument("--data"), argument("--baseline")
    expected_code = manifest["code_sha256"]
    essential = {"possession_lab/kinematics.py", "possession_lab/strong_motion_noise.py",
                 "possession_lab/robust.py", "possession_lab/probabilistic.py",
                 "possession_lab/probabilistic_run.py", "possession_lab/multimatch_run.py",
                 "possession_lab/data.py", "possession_lab/evaluation.py",
                 "evaluation/possession_metrics.py"}
    normalized_code = {name.replace("\\", "/"): digest for name, digest in expected_code.items()}
    if not essential <= normalized_code.keys():
        raise ValueError("C1 source manifest is incomplete")
    for relative, digest in normalized_code.items():
        path = (ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != digest:
            raise ValueError("Source differs from C1: "+relative)
    packages = {name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "python-crfsuite")}
    if packages != manifest["packages"]:
        raise ValueError("Package versions differ from C1; restore the C1 environment")
    expected_inputs = {normalized_path(path): digest
                       for path, digest in manifest["inputs_sha256"].items()}
    bundle = baseline/("leave_out_"+HELD)/"augmentation"
    marker = json_read(bundle/"complete.json")
    if set(marker["artifact_sha256"]) != set(ARTIFACTS):
        raise ValueError("Unexpected baseline artifact manifest")
    for name in ("complete.json", *ARTIFACTS):
        path = bundle/name
        if sha256(path) != expected_inputs.get(normalized_path(path)):
            raise ValueError("Baseline artifact differs from the completed C1 run: "+name)
    frozen = load_completed(bundle)
    config = frozen.config
    validate_baseline_config(config)
    protocol = json_read(c1_run/"experiment"/"protocol.json")
    if (protocol["held"] != HELD or protocol["fit_matches"] != SOURCES
            or protocol["eval"] != list(SCENARIOS) or protocol["eval_seeds"] != [1000, 8100]
            or protocol["baseline_hash"] != sha256(bundle/"model.crfsuite")):
        raise ValueError("C1 experiment protocol mismatch")
    return data, config, references, expected_inputs, {
        "c1_run": str(c1_run), "c1_manifest_sha256": sha256(manifest_path),
        "c1_scores_sha256": sha256(c1_run/"experiment"/"scores.json"),
        "c1_protocol_sha256": sha256(c1_run/"experiment"/"protocol.json"),
        "baseline": str(bundle), "baseline_sha256": sha256(bundle/"model.crfsuite"),
        "packages": packages, "c1_code_sha256": normalized_code,
        "historical_spell_source": "possession_lab/FOLLOWUP_2026-11-02.md",
        "historical_spell": HISTORICAL_SPELL,
    }


def evaluation_cases(halves):
    cases = {"clean": halves}
    cases["missing40"] = [(mask_ball(d, .4, 1000+i)[0], y)
                          for i, (d, y) in enumerate(halves)]
    cases["noise_0.3m"] = [(perturb(d, "noise_0.3m", 8100+i)[0], y)
                           for i, (d, y) in enumerate(halves)]
    cases["noise_plus_missing40"] = [(mask_ball(d, .4, 1000+i)[0], y)
                                     for i, (d, y) in enumerate(cases["noise_0.3m"])]
    return cases


def training_sequences(train, arm):
    views = [[], []]
    for data, truth, seed in train:
        for index, view in enumerate(train_views(data, seed, True)):
            views[index].append(make_sequence(view, truth, arm))
    # Exact C1 order: first view of ALL halves, then second view of ALL halves.
    return views[0] + views[1]


def calibration_sequences(cal, arm):
    return ([make_sequence(d, y, arm) for d, y, _ in cal] +
            [make_sequence(mask_ball(d, .4, seed)[0], y, arm) for d, y, seed in cal])


def fit_arm(train, cal, config, arm, folder, model_factory=RobustCRF):
    timings = {}
    start = time.perf_counter()
    seqs = training_sequences(train, arm)
    timings["training_features_s"] = time.perf_counter()-start
    start = time.perf_counter()
    model = model_factory(False)
    model.names = NAMES
    model.fit(seqs, folder)
    timings["crf_fit_s"] = time.perf_counter()-start
    del seqs
    start = time.perf_counter()
    cal_seqs = calibration_sequences(cal, arm)
    timings["calibration_features_s"] = time.perf_counter()-start
    start = time.perf_counter()
    raw = [model.predict(seq)[0] for seq in cal_seqs]
    truth = np.concatenate([seq.y for seq in cal_seqs])
    emission = fit_temperature(np.concatenate(raw), truth)
    trans, prior = np.asarray(config["transition"]), np.asarray(config["prior"])
    chained = [fusion([temperature(p, emission)], [1.], trans, prior, .25)[0] for p in raw]
    chain = fit_temperature(np.concatenate(chained), truth)
    timings["calibration_predict_and_fit_s"] = time.perf_counter()-start
    fitted = {"names": list(NAMES), "history": False, "variant": "motion_noise",
              "arm": arm, "velocity_backend": "five_frame" if arm == "A" else "ols_0.6s",
              "emission_temperature": emission, "chain_temperature": chain,
              "transition": trans.tolist(), "prior": prior.tolist(), "strength": .25,
              "stride": STRIDE, "training_matches": SOURCES, "partitions": config["partitions"],
              "crf": {"c1": 0, "c2": .1, "max_iterations": 80,
                      "feature.possible_transitions": True, "feature.possible_states": True},
              "context_samples": [-5, 0, 5], "acceleration": "five_frame_velocity_difference"}
    return model, fitted, timings


def pooled_availability(parts):
    totals = {}
    for part in parts:
        for grid, measures in part.items():
            for name, counts in measures.items():
                dest = totals.setdefault(grid, {}).setdefault(name,
                    {"valid": 0, "observed": 0, "total": 0})
                for field in dest:
                    dest[field] += counts[field]
    for measures in totals.values():
        for counts in measures.values():
            counts["fraction_all"] = counts["valid"]/counts["total"] if counts["total"] else None
            counts["fraction_observed"] = counts["valid"]/counts["observed"] if counts["observed"] else None
    return totals


def evaluate_arm(model, config, cases, arm, pm):
    rows = []
    trans, prior = np.asarray(config["transition"]), np.asarray(config["prior"])
    for scenario in SCENARIOS:
        start = time.perf_counter()
        scores, diagnostics, periods = [], [], []
        times = {"features_and_availability_s": 0., "predict_decode_s": 0., "scoring_s": 0.}
        for data, truth in cases[scenario]:
            tick = time.perf_counter()
            seq, counts = make_sequence(data, None, arm, diagnostics=True)
            times["features_and_availability_s"] += time.perf_counter()-tick
            tick = time.perf_counter()
            p = model.predict(seq)[0]
            # C1 fits chain_temperature but its scored path uses ONLY emission_temperature.
            path = fusion([temperature(p, config["emission_temperature"])],
                          [1.], trans, prior, .25)[1]
            prediction = np.repeat(path, STRIDE)[:len(truth)]
            times["predict_decode_s"] += time.perf_counter()-tick
            tick = time.perf_counter()
            score = evaluate_period(pm, truth, prediction)
            scores.append(score)
            times["scoring_s"] += time.perf_counter()-tick
            diagnostics.append(counts)
            periods.append({"period": data.period, "availability_counts": counts,
                            "scoring_counts": score,
                            "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest()})
        row = {"arm": arm, "variant": "motion_noise", "scenario": scenario,
               **aggregate(pm, scores), "availability": pooled_availability(diagnostics),
               "periods": periods, "timing": times,
               "elapsed_s": time.perf_counter()-start}
        rows.append(row)
    return rows


def comparisons(rows, reference):
    lookup = {(row["arm"], row["scenario"]): row for row in rows}
    if len(rows) != 8 or set(lookup) != {(arm, s) for arm in ("A", "B") for s in SCENARIOS}:
        raise ValueError("Expected exactly eight distinct A/B scenario scores")
    compared = []
    for scenario in SCENARIOS:
        for metric in METRICS:
            a, b, c1 = (float(row[metric]) for row in
                       (lookup["A", scenario], lookup["B", scenario], reference[scenario]))
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (a, b, c1)):
                raise ValueError("Invalid metric result")
            historical = HISTORICAL_SPELL[scenario] if metric == "spell_f1" else None
            compared.append({"scenario": scenario, "metric": metric, "C1": c1, "A": a, "B": b,
                "A_minus_C1": a-c1, "B_minus_A": b-a,
                "A_matches_C1_4dp": f"{a:.4f}" == f"{c1:.4f}",
                "historical_table": historical,
                "A_matches_table_4dp": None if historical is None else f"{a:.4f}" == f"{historical:.4f}"})
    return compared


def write_owner_report(out, rows, compared, timing, agreement):
    lines = ["# C2 DFL: motion_noise A/B on J03WMX", "",
             "A: original five-frame velocity. B: OLS over 16 observations / 0.6 s.",
             "35 features in both arms. Each arm fitted and calibrated separately.",
             "C1 reference: the completed owner run identified by hashes in protocol.json.",
             "Full metric/availability counts and per-half scoring: scores.json.",
             "Signed differences are fractions (multiply by 100 for percentage points).", "",
             f"A reference agreement at four decimals: **{agreement}**.",
             "A mismatch makes the B/A ablation provisional; investigate provenance first.", "",
             "| Scenario | Metric | C1 | A | B | A-C1 | B-A | A=C1 (4dp) |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in compared:
        lines.append("| "+" | ".join([row["scenario"], row["metric"]] +
            [f"{row[key]:.6f}" for key in ("C1", "A", "B", "A_minus_C1", "B_minus_A")] +
            [str(row["A_matches_C1_4dp"])])+" |")
    lines += ["", "Historical published motion_noise Spell F1: " +
              ", ".join(f"{s}={v:.4f}" for s, v in HISTORICAL_SPELL.items())+".",
              "Table agreement flags are also retained in comparison.csv.", "",
              "## Estimate availability", "",
              "Fractions below include startup/missing frames and inactive player slots.",
              "JSON also includes availability conditional on a finite current observation,",
              "both sampling grids, per-team counts and per-half counts.", "",
              "| Arm | Scenario | Grid | Ball v | Players v | Ball acceleration |",
              "|---|---|---|---:|---:|---:|"]
    for row in rows:
        for grid, counts in row["availability"].items():
            values = [counts[key]["fraction_all"] for key in
                      ("ball_velocity", "player_velocity", "ball_acceleration")]
            lines.append("| "+" | ".join([row["arm"], row["scenario"], grid]+
                         ["n/a" if x is None else f"{100*x:.2f}%" for x in values])+" |")
    lines += ["", "## Wall-clock timing (seconds)", "", "| Phase | Seconds |", "|---|---:|"]
    for phase, value in timing.items():
        lines.append(f"| {phase} | {value:.3f} |")
    lines += ["", "| Arm | Scenario | Features + availability | Predict/decode | Score | Total |",
              "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| "+" | ".join([row["arm"], row["scenario"]]+
            [f"{row['timing'][key]:.3f}" for key in
             ("features_and_availability_s", "predict_decode_s", "scoring_s")]+
            [f"{row['elapsed_s']:.3f}"])+" |")
    lines += ["", "A then B use the same precomputed evaluation observations. Timings are one",
              "sequential run, include diagnostic overhead, and are not repeated speed benchmarks.",
              "The 0.6 s velocity is causal; CRF +5 context and whole-sequence decoding are unchanged.",
              "This one development fold does not establish held-out generalization across matches.",
              "No result is omitted based on velocity availability; all original scoring frames remain.",
              "chain_temperature is fitted/saved for parity but does not alter C1's scored Viterbi path.", ""]
    (out/"REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(c1_run, output=None):
    branch = (ROOT/".git"/"HEAD").read_text().strip()
    if branch != "ref: refs/heads/codex/c1-sandbox":
        raise RuntimeError("Use codex/c1-sandbox; this runner never changes branches")
    if Path(sys.prefix).resolve() != ROOT/".venv":
        raise RuntimeError("Use the project .venv interpreter")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    out = safe_output(output if output is not None else PRIVATE_RUNS/("c2-dfl-J03WMX-"+tag))
    c1_run = Path(c1_run).resolve()
    start = time.perf_counter()
    data, config, reference, input_hashes, provenance = c1_inputs(c1_run)
    out.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "output": str(out), "python": sys.version, "executable": sys.executable,
              "branch": "codex/c1-sandbox", "held": HELD, "training_matches": SOURCES}
    json_write(out/"status.json", status)
    print("C2 output: "+str(out), flush=True)
    try:
        scripts = [HERE/name for name in ("run_c2_dfl.py", "c2_dfl_features.py", "c2_velocity.py")]
        protocol = {**provenance, "held": HELD, "sources": SOURCES, "names": list(NAMES),
                    "arms": {"A": "five_frame", "B": "ols_0.6s"},
                    "input_hz": 25, "stride": STRIDE, "scenarios": list(SCENARIOS),
                    "caps": {"ball_m_s": 60, "players_m_s": 15, "ball_accel_m_s2": 300},
                    "acceleration": "unchanged five-frame derivative of each arm's ball velocity",
                    "seeds": {"training": "7 + 10*match_index + half_index",
                              "noise_training_offset": 7200,
                              "calibration": "103 + 10*match_index + half_index",
                              "evaluation_missing": 1000, "evaluation_noise": 8100},
                    "augmentation": "unchanged train_views(..., noise=True); complementary 250-frame blocks",
                    "order": ["A", "B"], "reference_acceptance": "all 12 A/C1 metrics and 4 published Spell F1 cells equal at 4 decimals",
                    "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in scripts}}
        json_write(out/"protocol.json", protocol)
        timing = {"preflight_s": time.perf_counter()-start}
        tick = time.perf_counter()
        datasets, audits, files = {}, {}, []
        for match in MATCHES:
            print("READ "+match, flush=True)
            datasets[match], audits[match], paths = read_match(data, match)
            for record in paths:
                if record["sha256"] != input_hashes.get(normalized_path(record["path"])):
                    raise ValueError("Input differs from the completed C1 run: "+record["path"])
            files.extend(paths)
        timing["load_and_input_hashes_s"] = time.perf_counter()-tick
        json_write(out/"inputs.json", files)
        json_write(out/"audits.json", audits)
        train, cal, records = partitions(datasets, SOURCES)
        validate_partitions(records, config)
        json_write(out/"partitions.json", records)
        json_write(out/"source_seeds.json", {
            "train": [{"period": d.period, "seed": seed, "frames": len(y)} for d,y,seed in train],
            "calibration": [{"period": d.period, "seed": seed, "frames": len(y)} for d,y,seed in cal]})
        tick = time.perf_counter()
        cases = evaluation_cases(datasets[HELD])
        timing["evaluation_perturbations_s"] = time.perf_counter()-tick
        pm = official(ROOT)
        rows = []
        for arm in ("A", "B"):
            arm_start = time.perf_counter()
            print("FIT/CALIBRATE "+arm, flush=True)
            model, fitted, phases = fit_arm(train, cal, config, arm, out/arm)
            json_write(out/arm/"config.json", fitted)
            for phase, elapsed in phases.items():
                timing[arm+"_"+phase] = elapsed
            arm_rows = evaluate_arm(model, fitted, cases, arm, pm)
            rows.extend(arm_rows)
            timing[arm+"_total_s"] = time.perf_counter()-arm_start
            json_write(out/"partial_scores.json", rows)
            for row in arm_rows:
                print(arm+" "+row["scenario"]+" "+" ".join(
                    f"{metric}={row[metric]:.6f}" for metric in METRICS), flush=True)
            del model
            gc.collect()
        compared = comparisons(rows, reference)
        agreement = all(row["A_matches_C1_4dp"] and row["A_matches_table_4dp"] is not False
                        for row in compared)
        json_write(out/"scores.json", rows)
        with (out/"comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(compared[0]))
            writer.writeheader()
            writer.writerows(compared)
        # Recheck source integrity after fitting; never silently compare changed implementations.
        for relative, digest in {**provenance["c1_code_sha256"], **protocol["code_sha256"]}.items():
            if sha256(ROOT/relative) != digest:
                raise RuntimeError("Code changed during the experiment: "+relative)
        timing["total_before_final_report_s"] = time.perf_counter()-start
        json_write(out/"timing.json", timing)
        write_owner_report(out, rows, compared, timing, agreement)
        status.update(status="completed" if agreement else "completed_reference_mismatch",
                      scores=8, A_matches_C1=agreement,
                      output_artifact_sha256={
                          (Path(arm)/name).as_posix(): sha256(out/arm/name)
                          for arm in ("A","B") for name in ("model.crfsuite","scaler.npz","config.json")})
        print("C2_FINISHED - A matches C1: "+str(agreement)+"; "+str(out/"REPORT.md"), flush=True)
        return 0 if agreement else 2
    except BaseException as error:
        status.update(status="failed_or_interrupted", error=str(error))
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        json_write(out/"status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1-run", type=Path, required=True,
                        help="Completed C1 owner run with manifest.json and experiment/scores.json")
    parser.add_argument("--out", type=Path,
                        help="Optional NEW directory under tracking-physics-private/runs; default is unique")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    return run(args.c1_run, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
