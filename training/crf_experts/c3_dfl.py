"""C3 DFL core: explicit fold provenance, unchanged C2 mathematics, no input I/O on import."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pycrfsuite

import c2_dead_specialist as dead
import c2_dfl_features as features
import c2_frozen_noise as noise
import c2_no_team_motion as shared
import run_c2_dfl as c2
from possession_lab.data import sha256
from possession_lab.multimatch_run import MATCHES, partitions
from possession_lab.probabilistic import Standardizer, temperature, fit_temperature, transition_parameters, probability_metrics
from possession_lab.probabilistic_run import fusion
from possession_lab.robust import RobustCRF

VERSION = "c3-dfl-locked-v2-native-schema"
NAMES = tuple(features.NAMES)
VARIANTS = ("A", dead.VARIANT)
CELLS = tuple(shared.CELLS)
MODEL_FILES = ("config.json", "model.crfsuite", "scaler.npz")
KEYS = {"bias"} | {f"{o}:{n}" for o in (-5, 0, 5) for n in NAMES}
CRITERIA = {
    "spell_mean_min": .005, "positive_matches_min": 4,
    "positive_non_development_min": 3, "non_development_mean_strict_min": 0.,
    "clean_mean_min": -.01, "clean_match_min": -.03,
    "restart_mean_min": -.02, "restart_match_min": -.10,
    "inference_time_limit": None,
}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()

def recipe():
    return {"version": VERSION, "matches": list(MATCHES), "names": list(NAMES),
            "variants": list(VARIANTS), "cells": [list(c) for c in CELLS],
            "crf": dict(dead.CRF_PARAMS), "criteria": dict(CRITERIA),
            "velocity": "five_frame", "input_hz": 25, "stride": 2,
            "caps": [60., 15., 300.], "context_samples": [-5, 0, 5],
            "scaler_floor": .05, "scaler_clip": [-10., 10.],
            "augmentation": "C2 motion_noise complementary 250-frame views",
            "sigma": .3, "rho": .9, "noise_domain": 203301, "mask_domain": 203302,
            "calibration": "C2 source clean+masked40; T_A, chain_T_A, T_D",
            "decoder_strength": .25, "identity_atol": dead.ATOL, "identity_rtol": dead.RTOL,
            "metrica_later": {"fold": "leave_out_J03WMX", "games": [1, 2],
                "references": ["pass_intervals", "event_proxy"],
                "conditions": ["clean", "missing40", "long40"], "implemented": False}}

@dataclass(frozen=True)
class Fold:
    held: str
    run_id: str
    protocol_sha256: str

    def __post_init__(self):
        if self.held not in MATCHES or not self.run_id:
            raise ValueError("Invalid fold")
        if len(self.protocol_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.protocol_sha256):
            raise ValueError("Expected pinned protocol SHA-256")

    @property
    def sources(self):
        return tuple(m for m in MATCHES if m != self.held)

    @property
    def name(self):
        return "leave_out_" + self.held

    def record(self):
        return {"held": self.held, "sources": list(self.sources), "run_id": self.run_id,
                "protocol_sha256": self.protocol_sha256, "recipe_sha256": digest(recipe())}

def array_digest(array):
    a = np.asarray(array)
    return {"shape": list(a.shape), "dtype": a.dtype.str,
            "sha256": hashlib.sha256(a.tobytes()).hexdigest()}

def observation_record(d, y=None):
    result = {"period": d.period, "fps": d.fps, "pitch": list(d.pitch),
              "arrays": {name: array_digest(getattr(d, name))
                         for name in ("ball", "players", "frame_ids", "teams")},
              "track_contract": "stable player column within half; no cross-half history"}
    if y is not None:
        result["truth"] = array_digest(y)
    return result

def freeze_halves(halves):
    if len(halves) != 2 or [d.period for d, _ in halves] != ["firstHalf", "secondHalf"]:
        raise ValueError("Expected two separate halves")
    for d, y in halves:
        if (d.fps != 25 or np.any(np.diff(d.frame_ids) != 1) or len(y) != len(d.ball)
                or not np.isin(y, [0, 1, 2]).all()
                or d.ball.dtype.kind != "f" or d.players.dtype.kind != "f"
                or not np.isin(d.teams, [0, 1]).all()
                or not np.isfinite(d.pitch).all() or np.any(np.asarray(d.pitch) <= 0)):
            raise ValueError("Invalid DFL timeline/labels/units")
        for a in (d.ball, d.players, d.frame_ids, d.teams, y):
            a.setflags(write=False)
    return halves

def source_partitions(ctx, datasets):
    if set(datasets) != set(ctx.sources):
        raise ValueError("Exactly four source matches, never held-out, are required")
    for m in ctx.sources:
        freeze_halves(datasets[m])
    train, cal, records = partitions(datasets, list(ctx.sources))
    validate_partitions(ctx, records)
    lineage = {m: [observation_record(d, y) for d, y in datasets[m]] for m in ctx.sources}
    return train, cal, records, lineage

def validate_partitions(ctx, records):
    if len(records) != 8:
        raise ValueError("Expected eight source-half records")
    for r, (m, h) in zip(records, ((m, h) for m in ctx.sources for h in ("firstHalf", "secondHalf"))):
        n = r["calibrate"][1]
        # Exact integer boundaries, including the existing minimum-length validation.
        from possession_lab.probabilistic import split_ranges
        split = split_ranges(n)
        if (r["match"] != m or r["half"] != h or r["fit"] != list(split["refit"])
                or r["calibrate"] != list(split["calibrate"])
                or r["fit_frames"][1] - r["fit_frames"][0] != r["fit"][1] - 1
                or r["calibration_frames"][1] - r["calibration_frames"][0] != n - r["calibrate"][0] - 1
                or r["calibration_frames"][0] - r["fit_frames"][1] != 251):
            raise ValueError("Source partition/purge/order mismatch")

def clean_transitions(train):
    seqs = [features.make_sequence(d, y, "A") for d, y, _ in train]
    counts, priors = np.full((3, 3), .1), np.ones(3)
    for s in seqs:
        np.add.at(counts, (s.y[:-1], s.y[1:]), 1)
        priors += np.bincount(s.y, minlength=3)
    trans, prior = transition_parameters(seqs)
    return trans, prior, {"transition_pseudocounts": counts.tolist(),
                         "class_pseudocounts": priors.tolist(),
                         "source": "clean source-fit chunks, stride=2, no sequence joins"}

def common_config(ctx, records, lineage):
    return {"c3_version": VERSION, "fold": ctx.record(), "names": list(NAMES),
            "history": False, "input_hz": 25, "stride": 2,
            "context_samples": [-5, 0, 5], "velocity_backend": "five_frame",
            "acceleration": "five_frame_velocity_difference", "crf": dict(dead.CRF_PARAMS),
            "training_matches": list(ctx.sources), "partitions": records,
            "source_lineage": lineage, "source_lineage_sha256": digest(lineage)}

def validate_config(ctx, cfg, role, a_config=None, a_hashes=None):
    if role not in VARIANTS:
        raise ValueError("Unknown role")
    validate_native_schema(cfg.get("native_schema"), ("0", "1", "2") if role == "A" else dead.LABELS)
    fixed = {"c3_version": VERSION, "fold": ctx.record(), "names": list(NAMES),
             "history": False, "input_hz": 25, "stride": 2, "context_samples": [-5, 0, 5],
             "velocity_backend": "five_frame", "acceleration": "five_frame_velocity_difference",
             "crf": dict(dead.CRF_PARAMS), "training_matches": list(ctx.sources)}
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError("Fold/role/35-feature recipe mismatch")
    validate_partitions(ctx, cfg["partitions"])
    if (set(cfg["source_lineage"]) != set(ctx.sources)
            or digest(cfg["source_lineage"]) != cfg["source_lineage_sha256"]):
        raise ValueError("Source lineage mismatch")
    if role == "A":
        if cfg.get("arm") != "A" or cfg.get("variant") != "motion_noise" or cfg.get("strength") != .25:
            raise ValueError("A recipe mismatch")
        for k in ("emission_temperature", "chain_temperature"):
            if not np.isfinite(cfg[k]) or not .2 <= cfg[k] <= 10:
                raise ValueError("Invalid A temperature")
        t, p = np.asarray(cfg["transition"]), np.asarray(cfg["prior"])
        if (t.shape != (3, 3) or p.shape != (3,) or not np.isfinite(t).all()
                or not np.isfinite(p).all() or np.any(t <= 0) or np.any(p <= 0)
                or not np.allclose(t.sum(1), 1) or not np.isclose(p.sum(), 1)):
            raise ValueError("Invalid source transition/prior")
    else:
        if a_config is None or a_hashes is None:
            raise ValueError("Specialist requires its fold A")
        validate_config(ctx, a_config, "A")
        expected = {"variant": dead.VARIANT, "model_role": "binary_dead_specialist",
                    "labels": list(dead.LABELS), "label_meanings": dead.MEANINGS,
                    "training_variant": "motion_noise",
                    "reference_config_sha256": digest(a_config), "reference_A_artifact_sha256": a_hashes}
        if (any(cfg.get(k) != v for k, v in expected.items())
                or cfg["partitions"] != a_config["partitions"]
                or cfg["source_lineage"] != a_config["source_lineage"]
                or any(k in cfg for k in ("chain_temperature", "emission_temperature", "joint_temperature"))):
            raise ValueError("Specialist binding/temperature mismatch")
        td = cfg.get("binary_emission_temperature")
        if not isinstance(td, (int, float)) or not np.isfinite(td) or not .2 <= td <= 10:
            raise ValueError("Invalid specialist temperature")

def validate_native_schema(schema, labels):
    """Full input contract is distinct from CRFsuite's sparse exported attributes."""
    if not isinstance(schema, dict) or set(schema) != {
            "version", "input_attribute_names", "native_attribute_names", "labels"}:
        raise ValueError("Missing or invalid native schema manifest")
    if (schema["version"] != 1 or schema["input_attribute_names"] != sorted(KEYS)
            or schema["labels"] != sorted(labels)):
        raise ValueError("Wrong full 106-input schema or native labels")
    names = schema["native_attribute_names"]
    if (not isinstance(names, list) or not all(isinstance(n, str) for n in names)
            or names != sorted(set(names)) or "bias" not in names or not set(names) <= KEYS):
        raise ValueError("Invalid or unknown exported native attributes")


def check_native_schema(tagger, labels, expected=None):
    """Pin the fitted export, then demand exact agreement when loading it.

    CRFsuite may omit input attributes from its native model, including attributes
    with nonzero inputs. Do not change training to force 106 exported attributes.
    All 106 input keys and all 35 feature names remain fixed.
    """
    native_labels = tagger.labels()
    if len(native_labels) != len(labels) or set(native_labels) != set(labels):
        raise ValueError("Wrong native labels")
    schema = {"version": 1, "input_attribute_names": sorted(KEYS),
              "native_attribute_names": sorted(tagger.info().attributes),
              "labels": sorted(labels)}
    validate_native_schema(schema, labels)
    if expected is not None:
        validate_native_schema(expected, labels)
        if schema != expected:
            missing = sorted(set(expected["native_attribute_names"]) - set(schema["native_attribute_names"]))
            extra = sorted(set(schema["native_attribute_names"]) - set(expected["native_attribute_names"]))
            raise ValueError(f"Native schema differs from fitted manifest: missing={missing}, extra={extra}")
    return schema

def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

def artifact_hashes(folder):
    return {n: sha256(Path(folder) / n) for n in MODEL_FILES}

def load_model(ctx, folder, role, hashes, *, a_model=None, a_config=None, a_hashes=None,
               tagger_factory=None):
    folder = Path(folder)
    if set(hashes) != set(MODEL_FILES) or artifact_hashes(folder) != hashes:
        raise ValueError("Artifact hash mismatch")
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    validate_config(ctx, cfg, role, a_config, a_hashes)
    with np.load(folder / "scaler.npz", allow_pickle=False) as z:
        if set(z.files) != {"mean", "scale"}:
            raise ValueError("Wrong scaler fields")
        scaler = Standardizer()
        scaler.mean, scaler.scale = z["mean"].copy(), z["scale"].copy()
    dead.check_scaler(scaler)
    tagger = (tagger_factory or pycrfsuite.Tagger)()
    if tagger.open(str(folder / "model.crfsuite")) is False:
        raise ValueError("Cannot load CRF")
    check_native_schema(tagger, ("0", "1", "2") if role == "A" else dead.LABELS, cfg["native_schema"])
    if role == "A":
        model = noise.FrozenCRF(False)
        model.names = NAMES
    else:
        if a_model is None:
            raise ValueError("Missing fold A model")
        dead.compare_scalers(scaler, a_model.scaler)
        model = dead.BinaryCRF(frozen=True)
        model.native_labels = dead.check_labels(tagger)
    scaler.mean.setflags(write=False)
    scaler.scale.setflags(write=False)
    model.scaler, model.tagger = scaler, tagger
    return model, cfg

def identity(p, cfg):
    e = dead.compose(p, p[:, 2])
    a, b = dead.decode(p, cfg), dead.decode(e, cfg)
    record = {"max_abs_error": float(np.max(np.abs(e - p))),
              "path_mismatches": int(np.count_nonzero(a != b)),
              "atol": dead.ATOL, "rtol": dead.RTOL,
              "passed": bool(np.allclose(e, p, atol=dead.ATOL, rtol=dead.RTOL) and np.array_equal(a, b))}
    if not record["passed"]:
        raise dead.IdentityMismatch("Composition is not identity")
    return record

def fit_pair(ctx, datasets, folder, *, a_factory=None, d_factory=None, tagger_factory=None):
    """Only source dictionaries enter here. No held-out loader is available."""
    folder = Path(folder)
    if folder.exists():
        raise ValueError("No overwrite/resume of a fold")
    folder.mkdir(parents=True)
    train, cal, records, lineage = source_partitions(ctx, datasets)
    write_new_json(folder / "partitions.json", records)
    seeds = {"train": [s for _, _, s in train], "calibration": [s for _, _, s in cal]}
    write_new_json(folder / "seeds.json", seeds)
    times = {}
    start = time.perf_counter()
    trans, prior, stats = clean_transitions(train)
    times["transition_features_and_counts_s"] = time.perf_counter() - start
    write_new_json(folder / "transition_counts.json", stats)
    start = time.perf_counter()
    seqs = c2.training_sequences(train, "A")
    for s in seqs:
        dead.check_features(s)
    times["A_training_features_s"] = time.perf_counter() - start
    start = time.perf_counter()
    a = (a_factory or RobustCRF)(False)
    a.names = NAMES
    a.fit(seqs, folder / "A")
    native_a = check_native_schema(a.tagger, ("0", "1", "2"))
    times["A_crf_fit_s"] = time.perf_counter() - start
    del seqs
    start = time.perf_counter()
    cal_seqs = c2.calibration_sequences(cal, "A")
    times["A_calibration_features_s"] = time.perf_counter() - start
    start = time.perf_counter()
    raw = [a.predict(s)[0] for s in cal_seqs]
    truth = np.concatenate([s.y for s in cal_seqs])
    ta = fit_temperature(np.concatenate(raw), truth)
    chained = [fusion([temperature(p, ta)], [1.], trans, prior, .25)[0] for p in raw]
    tc = fit_temperature(np.concatenate(chained), truth)
    times["A_calibration_predict_and_fit_s"] = time.perf_counter() - start
    ac = {**common_config(ctx, records, lineage), "arm": "A", "variant": "motion_noise",
          "strength": .25, "transition": trans.tolist(), "prior": prior.tolist(),
          "emission_temperature": float(ta), "chain_temperature": float(tc), "native_schema": native_a}
    validate_config(ctx, ac, "A")
    write_new_json(folder / "A/config.json", ac)
    ah = artifact_hashes(folder / "A")
    loaded_a, ac2 = load_model(ctx, folder / "A", "A", ah, tagger_factory=tagger_factory)
    controls = []
    for i, (s, original) in enumerate(zip(cal_seqs, raw)):
        restored = loaded_a.predict(s)[0]
        if not np.allclose(original, restored, atol=dead.ATOL, rtol=dead.RTOL):
            raise ValueError("A round-trip changed probabilities")
        controls.append({"calibration_sequence": i,
                         **identity(dead.calibrated(restored, ta, 3), ac)})
    if ac2 != ac:
        raise ValueError("A round-trip changed config")
    write_new_json(folder / "identity_source.json", controls)
    write_new_json(folder / "A/complete.json", {"fold": ctx.record(), "role": "A",
                                              "artifact_sha256": ah, "source_identity_passed": True})
    a = loaded_a
    # Own binary scaler is fitted inside BinaryCRF and compared before Trainer creation.
    start = time.perf_counter()
    ds = dead.training_sequences(train)
    times["D_training_features_s"] = time.perf_counter() - start
    start = time.perf_counter()
    binary = (d_factory or dead.BinaryCRF)().fit(ds, folder / "dead_specialist", a.scaler)
    native_d = check_native_schema(binary.tagger, dead.LABELS)
    times["D_crf_fit_s"] = time.perf_counter() - start
    del ds
    start = time.perf_counter()
    dcals = dead.calibration_sequences(cal)
    times["D_calibration_features_s"] = time.perf_counter() - start
    start = time.perf_counter()
    draw = [binary.predict_proba(s) for s in dcals]
    td = fit_temperature(np.concatenate(draw), np.concatenate([s.y for s in dcals]))
    times["D_calibration_predict_and_fit_s"] = time.perf_counter() - start
    dc = {**common_config(ctx, records, lineage), "variant": dead.VARIANT,
          "model_role": "binary_dead_specialist", "labels": list(dead.LABELS),
          "label_meanings": dict(dead.MEANINGS), "training_variant": "motion_noise",
          "reference_config_sha256": digest(ac), "reference_A_artifact_sha256": ah,
          "binary_emission_temperature": float(td), "scaler_matches_A": binary.scaler_check,
          "native_schema": native_d}
    validate_config(ctx, dc, dead.VARIANT, ac, ah)
    write_new_json(folder / "dead_specialist/config.json", dc)
    dh = artifact_hashes(folder / "dead_specialist")
    loaded_d, dc2 = load_model(ctx, folder / "dead_specialist", dead.VARIANT, dh,
                              a_model=a, a_config=ac, a_hashes=ah, tagger_factory=tagger_factory)
    for s, original in zip(dcals, draw):
        if not np.allclose(original, loaded_d.predict_proba(s), atol=dead.ATOL, rtol=dead.RTOL):
            raise ValueError("D round-trip changed probabilities")
    if dc2 != dc or artifact_hashes(folder / "A") != ah:
        raise ValueError("Frozen fold A or specialist config changed")
    after = {m: [observation_record(d, y) for d, y in datasets[m]] for m in ctx.sources}
    if after != lineage:
        raise ValueError("Fit/calibration mutated source observations")
    write_new_json(folder / "dead_specialist/complete.json",
                   {"fold": ctx.record(), "role": dead.VARIANT, "artifact_sha256": dh,
                    "reference_A_artifact_sha256": ah})
    write_new_json(folder / "training_timings.json", times)
    return (a, ac, ah), (loaded_d, dc, dh), times


def boundary_details(pm, halves, predictions):
    """Keep concrete boundary pairings; never bridge reference gaps/halves."""
    detail = []
    for (d, truth), pred in zip(halves, predictions):
        runs = pm.valid_runs(truth)
        filled = pm.fill_abstentions(pred, runs)
        for lo, hi in runs:
            gt, pr = pm.transitions(truth, [(lo, hi)]), pm.transitions(filled, [(lo, hi)])
            for kind in ("stop", "restart"):
                def select(ts):
                    if kind == "stop":
                        return [(i, 0, 1) for i, a, b in ts if a < 2 and b == 2]
                    return [(i, 1, 0) for i, a, b in ts if a == 2 and b < 2]
                pairs, missed, extra = pm.match_transitions(select(gt), select(pr), 25, "pair")
                def point(t):
                    return {"index": int(t[0]), "frame_id": int(d.frame_ids[t[0]])}
                detail.append({"period": d.period, "valid_run": [int(lo), int(hi)], "kind": kind,
                               "matched": [{"truth": point(g), "prediction": point(p),
                                            "signed_error_s": (p[0]-g[0])/25} for g, p in pairs],
                               "missed": [point(t) for t in missed],
                               "spurious": [point(t) for t in extra]})
    return detail

def probability_diagnostics(p, labels):
    valid = (labels >= 0) & (labels < 3)
    if not valid.any():
        return {"support": 0, "metrics": None}
    return {"support": int(valid.sum()),
            "metrics": probability_metrics(p[valid], labels[valid], p[valid].argmax(1))}

def infer_half(model_a, cfg_a, data, *, model_d=None, cfg_d=None):
    """No truth argument: feature/prediction path cannot inspect labels."""
    timing = {}
    tick = time.perf_counter()
    seq, counts = features.make_sequence(data, None, "A", diagnostics=True)
    dead.check_features(seq)
    timing["features_s"] = time.perf_counter() - tick
    tick = time.perf_counter()
    raw_a = model_a.predict(seq)[0]
    p = dead.calibrated(raw_a, cfg_a["emission_temperature"], 3)
    timing["A_predict_and_temperature_s"] = time.perf_counter() - tick
    q, e = p[:, 2], p
    timing["D_predict_and_temperature_s"] = 0.
    timing["composition_s"] = 0.
    if model_d is not None:
        tick = time.perf_counter()
        q = dead.calibrated(model_d.predict_proba(seq),
                            cfg_d["binary_emission_temperature"], 2)[:, 1]
        timing["D_predict_and_temperature_s"] = time.perf_counter() - tick
        tick = time.perf_counter()
        e = dead.compose(p, q)
        timing["composition_s"] = time.perf_counter() - tick
    tick = time.perf_counter()
    pred = np.repeat(dead.decode(e, cfg_a), 2)[:len(data.frame_ids)]
    timing["decoder_and_expand_s"] = time.perf_counter() - tick
    timing["standalone_inference_s"] = sum(timing.values())
    arrays = {"frame_ids_25Hz": data.frame_ids, "prediction_25Hz": pred,
              "p_A_12_5Hz": p, "q_D_12_5Hz": q, "emissions_12_5Hz": e}
    return arrays, counts, array_digest(seq.x), timing

def selected_mask(original, family, seed, condition, half_index):
    if "missing40" not in condition:
        return np.zeros(len(original.ball), dtype=bool)
    if family == "seeded":
        return noise.block_mask(original, seed, half_index)[0]
    from dataclasses import replace
    zero = replace(original, ball=np.zeros_like(original.ball))
    return ~np.isfinite(c2.mask_ball(zero, .4, 1000 + half_index)[0].ball).all(1)

def evaluate_pair(ctx, a_bundle, d_bundle, halves, pm, *, artifact_callback=None, on_rows=None):
    a, ac, ah = a_bundle
    d, dc, _ = d_bundle
    validate_config(ctx, ac, "A")
    validate_config(ctx, dc, dead.VARIANT, ac, ah)
    freeze_halves(halves)
    originals = [observation_record(x, y) for x, y in halves]
    original_state = dead.reference_state(a, ac)
    rows, identities, descriptors = [], [], []
    for family, seed, conditions, mapped, meta in shared.case_batches(halves):
        descriptors.append(shared.descriptor(family, seed, conditions, mapped, meta))
        for old_name, condition in zip(c2.SCENARIOS, conditions):
            observations = freeze_halves(mapped[old_name])
            before = [observation_record(x, y) for x, y in observations]
            predictions, scores, periods = {v: [] for v in VARIANTS}, {v: [] for v in VARIANTS}, {v: [] for v in VARIANTS}
            costs = {v: {} for v in VARIANTS}
            for h, (data, truth) in enumerate(observations):
                a_counts = a_features = None
                for variant in VARIANTS:
                    arrays, counts, feature_hash, timing = infer_half(
                        a, ac, data, model_d=d if variant != "A" else None, cfg_d=dc)
                    if variant == "A":
                        a_counts, a_features = counts, feature_hash
                        tick = time.perf_counter()
                        identities.append({"held": ctx.held, "family": family, "seed": seed,
                                           "condition": condition, "period": data.period,
                                           **identity(arrays["p_A_12_5Hz"], ac)})
                        timing["identity_control_s"] = time.perf_counter() - tick
                    elif counts != a_counts or feature_hash != a_features:
                        raise ValueError("Paired features/availability differ")
                    if observation_record(data, truth) != before[h]:
                        raise ValueError("A/B inference mutated shared observations")
                    tick = time.perf_counter()
                    prediction = arrays["prediction_25Hz"]
                    score = c2.evaluate_period(pm, truth, prediction)
                    scores[variant].append(score)
                    predictions[variant].append(prediction)
                    pd = {key: probability_diagnostics(arrays[key], truth[::2])
                          for key in ("p_A_12_5Hz", "emissions_12_5Hz")}
                    timing["scoring_s"] = time.perf_counter() - tick
                    period = {"period": data.period, "availability_counts": counts,
                              "scoring_counts": score, "observations": before[h],
                              "feature_sha256": feature_hash,
                              "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
                              "probability_metrics": pd}
                    tick = time.perf_counter()
                    bv, pv, ba = features.derivatives(data, "A")
                    mask = selected_mask(halves[h][0], family, seed, condition, h)
                    observed = np.isfinite(data.ball).all(1)
                    natural = ~np.isfinite(halves[h][0].ball).all(1)
                    arrays.update(ball_velocity_valid_25Hz=np.isfinite(bv).all(-1),
                                  player_velocity_valid_25Hz=np.isfinite(pv).all(-1),
                                  ball_acceleration_valid_25Hz=np.isfinite(ba).all(-1),
                                  observed_ball_25Hz=observed, selected_mask_25Hz=mask,
                                  natural_missing_ball_25Hz=natural)
                    period["coverage"] = {"frames": len(truth),
                        "graded_frames": int(np.count_nonzero(truth < 3)),
                        "predicted_frames": len(prediction), "natural_missing_frames": int(natural.sum()),
                        "selected_mask_frames": int(mask.sum()), "actual_missing_frames": int((~observed).sum()),
                        "added_missing_observed_frames": int((~natural & ~observed).sum()),
                        "selected_mask_sha256": array_digest(mask)}
                    timing["diagnostic_masks_s"] = time.perf_counter() - tick
                    tick = time.perf_counter()
                    if artifact_callback:
                        period["probability_artifact"] = artifact_callback(
                            ctx, family, seed, condition, variant, data, arrays)
                    timing["artifact_io_s"] = time.perf_counter() - tick
                    period["timing"] = timing
                    periods[variant].append(period)
                    for k, value in timing.items():
                        costs[variant][k] = costs[variant].get(k, 0.) + value
            for variant in VARIANTS:
                tick = time.perf_counter()
                diag = dead.dead_diagnostics(pm, observations, predictions[variant])
                detail = boundary_details(pm, observations, predictions[variant])
                costs[variant]["dead_diagnostics_s"] = time.perf_counter() - tick
                n = sum(len(y) for _, y in observations)
                total = costs[variant]["standalone_inference_s"]
                costs[variant].update(frames=n, fps=n/total if total else None,
                                      seconds_per_match=total,
                                      cache_policy="independent full variant paths, no feature/A cache")
                rows.append({"held": ctx.held, "fold": ctx.name, "family": family, "seed": seed,
                             "condition": condition, "variant": variant, "arm": variant, "complete": True,
                             **c2.aggregate(pm, scores[variant]),
                             "availability": c2.pooled_availability(
                                 [p["availability_counts"] for p in periods[variant]]),
                             "periods": periods[variant], "dead_diagnostics": diag,
                             "boundary_details": detail, "timing": costs[variant]})
            if [observation_record(x, y) for x, y in observations] != before:
                raise ValueError("Shared observations mutated")
            if on_rows:
                on_rows(rows)
    if (len(rows) != 48 or len(identities) != 48
            or [observation_record(x, y) for x, y in halves] != originals
            or dead.reference_state(a, ac) != original_state):
        raise ValueError("Incomplete fold or mutated frozen A/input")
    return rows, identities, descriptors

def require_legacy_gate(gate):
    if (gate.get("passed") is not True or gate.get("A_checks") != 197
            or gate.get("identity_checks") != 48 or gate.get("specialist_rows") != 24
            or gate.get("specialist_passed") is not True):
        raise ValueError("Legacy replay must complete before ANY source fit")

def run_folds(contexts, gate, source_loader, fit_callback, held_loader, evaluate_callback):
    """The only training dispatch. Five contexts; no per-seed fitting loop."""
    require_legacy_gate(gate)
    if len(contexts) != 5 or tuple(c.held for c in contexts) != tuple(MATCHES):
        raise ValueError("Expected five ordered unique folds")
    rows, fits = [], []
    for ctx in contexts:
        datasets = {m: source_loader(m) for m in ctx.sources}
        pair = fit_callback(ctx, datasets)
        fits.append({"held": ctx.held, "A_fits": 1, "specialist_fits": 1})
        del datasets
        held = held_loader(ctx.held)
        rows.extend(evaluate_callback(ctx, pair, held))
    if len(rows) != 240:
        raise ValueError("Incomplete evaluation matrix")
    return rows, fits
