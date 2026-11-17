"""Frozen C2 inference, paired IID/AR(1) perturbations, replay and summaries.

No fitting or calibration entry points are used. Existing features/scoring are reused.
"""
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pycrfsuite
from scipy.signal import lfilter

import c2_dfl_features as features
import run_c2_dfl as previous
from possession_lab.data import sha256
from possession_lab.probabilistic import Standardizer
from possession_lab.robust import RobustCRF, mask_ball

SEEDS = (2027, 2028, 2029, 2030, 2031)
CONDITIONS = ("iid", "ar1", "iid_missing40", "ar1_missing40")
SIGMA = 0.3
RHO = 0.9
NOISE_DOMAIN = 203301
MASK_DOMAIN = 203302
METRICS = ("spell_f1", "macro_f1", "f1_D")
MODEL_FILES = ("config.json", "scaler.npz", "model.crfsuite")


class FrozenCRF(RobustCRF):
    def fit(self, *args, **kwargs):
        raise RuntimeError("Frozen evaluation cannot train a model")


def validate_config(config, arm):
    expected = {
        "arm": arm, "history": False, "variant": "motion_noise", "stride": 2,
        "velocity_backend": {"A": "five_frame", "B": "ols_0.6s"}.get(arm),
        "acceleration": "five_frame_velocity_difference", "strength": .25,
        "context_samples": [-5, 0, 5], "training_matches": previous.SOURCES,
        "crf": {"c1": 0, "c2": .1, "max_iterations": 80,
                "feature.possible_transitions": True, "feature.possible_states": True},
    }
    if arm not in ("A", "B") or any(config.get(k) != v for k,v in expected.items()):
        raise ValueError("Saved arm configuration differs from the frozen C2 recipe")
    if config.get("names") != list(features.NAMES) or len(config["names"]) != 35:
        raise ValueError("Saved names must preserve all 35 feature names in the original order")
    for key in ("emission_temperature", "chain_temperature"):
        if not math.isfinite(config[key]) or not .2 <= config[key] <= 10:
            raise ValueError("Invalid saved temperature")
    transition, prior = np.asarray(config["transition"]), np.asarray(config["prior"])
    if (transition.shape != (3,3) or prior.shape != (3,)
            or not np.isfinite(transition).all() or not np.isfinite(prior).all()
            or np.any(transition <= 0) or np.any(prior <= 0)
            or not np.allclose(transition.sum(1), 1) or not np.isclose(prior.sum(), 1)):
        raise ValueError("Invalid saved transition/prior")


def assemble_model(config, arm, mean, scale, tagger):
    validate_config(config, arm)
    mean, scale = np.array(mean, copy=True), np.array(scale, copy=True)
    if (mean.shape != (35,) or scale.shape != (35,)
            or mean.dtype.kind != "f" or scale.dtype.kind != "f"
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale < .05)):
        raise ValueError("Expected finite 35-column scaler with the original scale floor")
    if set(tagger.labels()) != {"0", "1", "2"}:
        raise ValueError("Saved CRF must have H/A/D labels 0/1/2")
    model = FrozenCRF(False)
    # Restore the saved 35-name schema BEFORE associating it with the scaler.
    model.names = tuple(config["names"])
    model.scaler = Standardizer()
    mean.setflags(write=False)
    scale.setflags(write=False)
    model.scaler.mean, model.scaler.scale = mean, scale
    model.tagger = tagger
    return model


def load_frozen_arm(folder, arm, expected_hashes, *, tagger_factory=pycrfsuite.Tagger):
    """Read-only loader; expected hashes come from the previous C2 status.json."""
    folder = Path(folder)
    if set(expected_hashes) != set(MODEL_FILES):
        raise ValueError("Missing or unexpected saved-model hashes")
    for name in MODEL_FILES:
        if sha256(folder/name) != expected_hashes[name]:
            raise ValueError("Saved artifact hash mismatch: "+arm+"/"+name)
    config = json.loads((folder/"config.json").read_text(encoding="utf-8-sig"))
    validate_config(config, arm)
    with np.load(folder/"scaler.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"mean", "scale"}:
            raise ValueError("Unexpected scaler fields")
        mean, scale = archive["mean"].copy(), archive["scale"].copy()
    tagger = tagger_factory()
    if tagger.open(str(folder/"model.crfsuite")) is False:
        raise ValueError("Could not open the saved CRF")
    return assemble_model(config, arm, mean, scale, tagger), config


def paired_errors(shape, seed, half_index, track_ids=None, *, sigma=SIGMA, rho=RHO, reset_frames=None):
    """Shared standard-normal innovations, independent coordinate/track segments.

    The first dimension is a contiguous timeline. Gaps do not reset a known
    track. Track-ID changes reset the process to a fresh stationary initial draw.
    """
    if len(shape) != 3 or shape[-1] != 2 or sigma < 0 or not np.isfinite(sigma) or not 0 <= rho < 1:
        raise ValueError("Expected (frames, tracks, 2), finite sigma >= 0 and 0 <= rho < 1")
    n, tracks, _ = shape
    ids = np.broadcast_to(np.arange(tracks), (n, tracks)) if track_ids is None else np.asarray(track_ids)
    if ids.shape != (n, tracks) or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("Known nonnegative integer track IDs are required")
    resets = np.zeros((n, tracks), dtype=bool) if reset_frames is None else np.asarray(reset_frames)
    if resets.shape != (n, tracks) or resets.dtype.kind != "b":
        raise ValueError("reset_frames must be a boolean observation-shaped boundary mask")
    iid, ar = np.empty(shape, dtype=np.float64), np.empty(shape, dtype=np.float64)
    for slot in range(tracks):
        boundary = (ids[1:,slot] != ids[:-1,slot]) | resets[1:,slot]
        cuts = np.r_[0, np.flatnonzero(boundary)+1, n]
        for segment, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:])):
            if start == stop:
                continue
            for axis in (0, 1):
                state = np.random.SeedSequence([int(seed), NOISE_DOMAIN, int(half_index), slot, segment, axis])
                z = np.random.Generator(np.random.PCG64(state)).standard_normal(stop-start)
                iid[start:stop,slot,axis] = sigma*z
                ar[start,slot,axis] = sigma*z[0]
                if stop-start > 1:
                    tail, _ = lfilter([sigma*math.sqrt(1-rho*rho)], [1., -rho], z[1:],
                                      zi=[rho*ar[start,slot,axis]])
                    ar[start+1:stop,slot,axis] = tail
    return iid, ar


def paired_observations(data, seed, half_index, track_ids=None):
    """No labels accepted. Keep dtype and every nonfinite coordinate unchanged."""
    if data.fps != 25 or np.any(np.diff(data.frame_ids) != 1):
        raise ValueError("Noise requires the original contiguous 25 Hz input")
    points = np.concatenate((data.ball[:,None,:], data.players), axis=1)
    if points.dtype.kind != "f":
        raise ValueError("Floating-point tracking positions required")
    finite = np.isfinite(points)
    # Stable DFL columns may begin with absent rows. Start the observable track
    # with a fresh stationary draw; later same-track gaps do not trigger resets.
    observed = finite.any(axis=-1)
    resets = np.zeros(observed.shape, dtype=bool)
    for slot in np.flatnonzero(observed.any(axis=0)):
        resets[np.flatnonzero(observed[:,slot])[0],slot] = True
    iid, ar = paired_errors(points.shape, seed, half_index, track_ids, reset_frames=resets)
    views = []
    for error in (iid, ar):
        changed = points.copy()
        # Match C1's in-place casting back to the input coordinate dtype.
        changed[finite] += error[finite]
        views.append(replace(data, ball=changed[:,0,:], players=changed[:,1:,:]))
    return tuple(views)


def block_mask(data, seed, half_index):
    state = np.random.SeedSequence([int(seed), MASK_DOMAIN, int(half_index)])
    mask_seed = int(state.generate_state(1, dtype=np.uint32)[0])
    # A zero ball exposes the sampled mask even where the original ball is missing.
    dummy = replace(data, ball=np.zeros_like(data.ball))
    masked, rate = mask_ball(dummy, .4, mask_seed)
    mask = ~np.isfinite(masked.ball).all(1)
    return mask, mask_seed, rate


def fingerprint(data):
    return hashlib.sha256(data.ball.tobytes()+data.players.tobytes()+data.frame_ids.tobytes()).hexdigest()


def paired_cases(halves, seed):
    cases = {name: [] for name in CONDITIONS}
    metadata = []
    for half_index, (data, truth) in enumerate(halves):
        iid, ar = paired_observations(data, seed, half_index)
        mask, mask_seed, rate = block_mask(data, seed, half_index)
        masked = []
        for view in (iid, ar):
            ball = view.ball.copy()
            ball[mask] = np.nan
            masked.append(replace(view, ball=ball))
        record = {"period": data.period, "half_index": half_index, "mask_seed": mask_seed,
                  "selected_mask_fraction": rate, "selected_mask_frames": int(mask.sum()),
                  "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(), "observations_sha256": {}}
        for condition, view in zip(CONDITIONS, (iid, ar, *masked)):
            view.ball.setflags(write=False)
            view.players.setflags(write=False)
            cases[condition].append((view, truth))
            record["observations_sha256"][condition] = fingerprint(view)
        metadata.append(record)
    return cases, metadata


def recorded_precision_match(actual, recorded):
    """Round only to the decimal precision encoded in the saved scores.json token."""
    if isinstance(recorded, bool) or not isinstance(recorded, (int, float, Decimal)):
        return False
    expected = recorded if isinstance(recorded, Decimal) else Decimal(str(recorded))
    if not expected.is_finite() or not 0 <= expected <= 1 or not math.isfinite(actual) or not 0 <= actual <= 1:
        return False
    with localcontext() as context:
        context.prec = max(60, len(expected.as_tuple().digits)+abs(expected.as_tuple().exponent)+5)
        unit = Decimal(1).scaleb(expected.as_tuple().exponent)
        return Decimal(str(actual)).quantize(unit) == expected


def replay_checks(rows, reference):
    expected = {(arm, scenario) for arm in ("A","B") for scenario in previous.SCENARIOS}
    def index(values):
        keys = [(row["arm"],row["scenario"]) for row in values]
        if len(keys) != 8 or set(keys) != expected:
            raise ValueError("Replay requires eight distinct original A/B scenario rows")
        return dict(zip(keys, values))
    observed, saved = index(rows), index(reference)
    checks = []
    for arm, scenario in sorted(expected):
        actual, target = observed[arm,scenario], saved[arm,scenario]
        for metric in METRICS:
            checks.append({"arm":arm,"scenario":scenario,"check":metric,
                           "recorded_token":str(target[metric]),"actual":actual[metric],
                           "passed":recorded_precision_match(actual[metric],target[metric])})
        checks.append({"arm":arm,"scenario":scenario,"check":"confusion",
                       "passed":actual["confusion"]==target["confusion"]})
        if len(actual["periods"]) != 2 or len(target["periods"]) != 2:
            raise ValueError("Replay must preserve the two separate halves")
        for a,b in zip(actual["periods"],target["periods"]):
            same_period = a["period"]==b["period"]
            for field in ("prediction_sha256","availability_counts"):
                checks.append({"arm":arm,"scenario":scenario,"period":a["period"],"check":field,
                               "passed":same_period and a[field]==b[field]})
    return checks


class ReplayMismatch(RuntimeError):
    pass


def evaluate_frozen(models, configs, halves, reference, pm, *, on_replay=None, on_batch=None):
    original_cases = previous.evaluation_cases(halves)
    replay_rows = []
    for arm in ("A","B"):
        replay_rows.extend(previous.evaluate_arm(models[arm],configs[arm],original_cases,arm,pm))
    checks = replay_checks(replay_rows,reference)
    if on_replay:
        on_replay(replay_rows,checks)
    if not all(check["passed"] for check in checks):
        raise ReplayMismatch("Original scenarios do not reproduce; new evaluation was NOT started")
    del original_cases
    rows, perturbations = [], []
    for seed in SEEDS:
        cases, metadata = paired_cases(halves,seed)
        # Reuse the unchanged scorer/evaluator; its four dictionary keys are labels only.
        mapped = {old: cases[new] for old,new in zip(previous.SCENARIOS,CONDITIONS)}
        batch = []
        for arm in ("A","B"):
            arm_rows = previous.evaluate_arm(models[arm],configs[arm],mapped,arm,pm)
            for row,condition in zip(arm_rows,CONDITIONS):
                row.pop("scenario")
                row.update(seed=seed,condition=condition)
                batch.append(row)
        rows.extend(batch)
        perturbations.append({"seed":seed,"halves":metadata})
        if on_batch:
            on_batch(rows,perturbations)
    return rows,perturbations,checks


def descriptive(values):
    defined = [value for value in values if value is not None]
    result = {"n":len(values),"n_defined":len(defined),"mean":None,"sample_sd":None,"min":None,"max":None}
    if defined and len(defined)==len(values):
        a=np.asarray(defined,dtype=float)
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite descriptive statistic")
        result.update(mean=float(a.mean()),sample_sd=float(a.std(ddof=1)) if len(a)>1 else None,
                      min=float(a.min()),max=float(a.max()))
    return result


def comparisons_and_summary(rows):
    expected={(seed,condition,arm) for seed in SEEDS for condition in CONDITIONS for arm in ("A","B")}
    keys=[(row["seed"],row["condition"],row["arm"]) for row in rows]
    if len(keys)!=40 or set(keys)!=expected:
        raise ValueError("All five seeds, four conditions and both arms are required")
    lookup=dict(zip(keys,rows))
    paired, availability=[],[]
    for seed in SEEDS:
        for condition in CONDITIONS:
            a,b=lookup[seed,condition,"A"],lookup[seed,condition,"B"]
            for metric in METRICS:
                x,y=float(a[metric]),float(b[metric])
                if any(not math.isfinite(v) or not 0<=v<=1 for v in (x,y)):
                    raise ValueError("Invalid F1 value")
                paired.append({"seed":seed,"condition":condition,"metric":metric,"A":x,"B":y,
                               "B_minus_A":y-x,"B_minus_A_pp":100*(y-x)})
            for grid in ("25Hz","12.5Hz"):
                for measure in ("ball_velocity","player_velocity","ball_acceleration","home_velocity","away_velocity"):
                    ca,cb=a["availability"][grid][measure],b["availability"][grid][measure]
                    row={"seed":seed,"condition":condition,"grid":grid,"measure":measure}
                    for arm,counts in (("A",ca),("B",cb)):
                        row.update({arm+"_"+key:value for key,value in counts.items()})
                    for fraction in ("fraction_all","fraction_observed"):
                        x,y=ca[fraction],cb[fraction]
                        row["B_minus_A_"+fraction+"_pp"]=None if x is None or y is None else 100*(y-x)
                    availability.append(row)
    summary=[]
    for condition in CONDITIONS:
        for metric in METRICS:
            group=[row for row in paired if row["condition"]==condition and row["metric"]==metric]
            summary.append({"condition":condition,"metric":metric,
                            **{role:descriptive([row[role] for row in group]) for role in ("A","B","B_minus_A")}})
    coverage_summary=[]
    for condition in CONDITIONS:
        for grid in ("25Hz","12.5Hz"):
            for measure in ("ball_velocity","player_velocity","ball_acceleration","home_velocity","away_velocity"):
                group=[r for r in availability if (r["condition"],r["grid"],r["measure"])==(condition,grid,measure)]
                fields=("A_fraction_all","B_fraction_all","B_minus_A_fraction_all_pp",
                        "A_fraction_observed","B_fraction_observed","B_minus_A_fraction_observed_pp")
                coverage_summary.append({"condition":condition,"grid":grid,"measure":measure,
                                        **{field:descriptive([r[field] for r in group]) for field in fields}})
    return paired,availability,summary,coverage_summary
