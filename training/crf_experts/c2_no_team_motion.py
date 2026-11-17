"""One C2 ablation: delete six team aggregates from A's model inputs.

The existing 25 Hz derivative/feature adapter, generators and scorer are unchanged.
"""
from dataclasses import replace
from pathlib import Path
import copy
import hashlib
import json
import math
import time
import numpy as np
import pycrfsuite

import c2_dfl_features as features
import c2_frozen_noise as noise
import run_c2_dfl as previous
from possession_lab.data import sha256
from possession_lab.probabilistic import Standardizer, temperature, fit_temperature
from possession_lab.probabilistic_run import fusion
from possession_lab.robust import RobustCRF, mask_ball

VARIANT = "A_no_team_motion"
FULL_NAMES = tuple(features.NAMES)
REMOVED = ("H_speed_mean", "H_collective_speed", "H_motion_coherence",
           "A_speed_mean", "A_collective_speed", "A_motion_coherence")
KEEP = tuple(i for i,name in enumerate(FULL_NAMES) if name not in REMOVED)
NAMES = tuple(FULL_NAMES[i] for i in KEEP)
PROJECTION = {"input_names":list(FULL_NAMES),"removed_names":list(REMOVED),"keep_indices":list(KEEP)}
if len(FULL_NAMES)!=35 or len(NAMES)!=29 or tuple(FULL_NAMES.index(n) for n in REMOVED)!=(17,18,19,26,27,28):
    raise RuntimeError("Unexpected source feature schema; do not adapt the experiment silently")
CELLS = tuple([("historical",None,s) for s in previous.SCENARIOS] +
              [("seeded",seed,s) for seed in noise.SEEDS for s in noise.CONDITIONS])


def project_sequence(seq):
    if seq.x.ndim!=2 or seq.x.shape[1]!=35:
        raise ValueError("Projection requires exactly the original 35 columns")
    return replace(seq,x=seq.x[:,KEEP].copy())


def require_schema(model):
    if (tuple(model.names)!=NAMES or model.scaler.mean.shape!=(29,)
            or model.scaler.scale.shape!=(29,)):
        raise ValueError("A_no_team_motion must use its own 29-feature model and scaler")


class ProjectedPredictor:
    """Unchanged evaluator supplies A's 35 columns; actual model sees only 29."""
    def __init__(self,model):
        require_schema(model)
        self.model=model
    def predict(self,seq):
        return self.model.predict(project_sequence(seq))


def source_partitions(datasets,reference_config):
    if set(datasets)!=set(previous.SOURCES) or reference_config["training_matches"]!=previous.SOURCES:
        raise ValueError("Only the four source matches may enter fit/calibration")
    train,cal,records=previous.partitions(datasets,previous.SOURCES)
    previous.validate_partitions(records,reference_config)
    return train,cal,records


def training_sequences(train):
    views=[[],[]]
    for data,truth,seed in train:
        for index,view in enumerate(previous.train_views(data,seed,True)):
            views[index].append(project_sequence(features.make_sequence(view,truth,"A")))
    return views[0]+views[1]


def calibration_sequences(cal):
    clean=[project_sequence(features.make_sequence(d,y,"A")) for d,y,_ in cal]
    masked=[project_sequence(features.make_sequence(mask_ball(d,.4,seed)[0],y,"A")) for d,y,seed in cal]
    return clean+masked


def fit_variant(train,cal,reference_config,folder,*,model_factory=RobustCRF):
    noise.validate_config(reference_config,"A")
    times={}
    start=time.perf_counter()
    seqs=training_sequences(train)
    times["training_features_s"]=time.perf_counter()-start
    start=time.perf_counter()
    model=model_factory(False)
    model.names=NAMES
    model.fit(seqs,folder)
    require_schema(model)
    times["crf_fit_s"]=time.perf_counter()-start
    del seqs
    start=time.perf_counter()
    seqs=calibration_sequences(cal)
    times["calibration_features_s"]=time.perf_counter()-start
    start=time.perf_counter()
    raw=[model.predict(seq)[0] for seq in seqs]
    truth=np.concatenate([seq.y for seq in seqs])
    emission=fit_temperature(np.concatenate(raw),truth)
    trans,prior=np.asarray(reference_config["transition"]),np.asarray(reference_config["prior"])
    chained=[fusion([temperature(p,emission)],[1.],trans,prior,.25)[0] for p in raw]
    chain=fit_temperature(np.concatenate(chained),truth)
    times["calibration_predict_and_fit_s"]=time.perf_counter()-start
    config=copy.deepcopy(reference_config)
    config.update(arm=VARIANT,variant=VARIANT,names=list(NAMES),
                  reference_variant="A",training_variant="motion_noise",
                  feature_projection=copy.deepcopy(PROJECTION),
                  emission_temperature=emission,chain_temperature=chain)
    return model,config,times


def validate_variant_config(config):
    if (config.get("arm")!=VARIANT or config.get("variant")!=VARIANT
            or config.get("names")!=list(NAMES) or config.get("feature_projection")!=PROJECTION
            or config.get("reference_variant")!="A" or config.get("training_variant")!="motion_noise"):
        raise ValueError("Invalid saved 29-feature ablation schema")
    # Validate unchanged settings without altering the saved configuration.
    base={**config,"arm":"A","variant":"motion_noise","names":list(FULL_NAMES)}
    noise.validate_config(base,"A")


def load_variant(folder,expected_hashes,*,tagger_factory=pycrfsuite.Tagger):
    folder=Path(folder)
    if set(expected_hashes)!=set(noise.MODEL_FILES):
        raise ValueError("Expected hashes of the three new model artifacts")
    for name in noise.MODEL_FILES:
        if sha256(folder/name)!=expected_hashes[name]:
            raise ValueError("New variant artifact hash mismatch: "+name)
    config=json.loads((folder/"config.json").read_text(encoding="utf-8-sig"))
    validate_variant_config(config)
    with np.load(folder/"scaler.npz",allow_pickle=False) as archive:
        if set(archive.files)!={"mean","scale"}:
            raise ValueError("Unexpected scaler fields")
        mean,scale=archive["mean"].copy(),archive["scale"].copy()
    if (mean.shape!=(29,) or scale.shape!=(29,) or mean.dtype.kind!="f" or scale.dtype.kind!="f"
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<.05)):
        raise ValueError("Expected a finite independently fitted 29-column scaler")
    model=noise.FrozenCRF(False)
    model.names=tuple(config["names"])
    model.scaler=Standardizer()
    mean.setflags(write=False);scale.setflags(write=False)
    model.scaler.mean,model.scaler.scale=mean,scale
    model.tagger=tagger_factory()
    if model.tagger.open(str(folder/"model.crfsuite")) is False or set(model.tagger.labels())!={"0","1","2"}:
        raise ValueError("Invalid saved 29-feature CRF")
    require_schema(model)
    return model,config


def cell(row):
    return row["family"],row["seed"],row["condition"]


def normalize_rows(rows,family,seed,conditions,variant):
    normalized=[]
    for row,condition in zip(rows,conditions):
        result={k:v for k,v in row.items() if k!="scenario"}
        result.update(family=family,seed=seed,condition=condition,arm=variant,variant=variant)
        normalized.append(result)
    if len(normalized)!=4 or len(rows)!=4:
        raise ValueError("Expected exactly four conditions per evaluation batch")
    return normalized


def reference_rows(original,frozen):
    expected_original={(a,s) for a in ("A","B") for s in previous.SCENARIOS}
    keys=[(r["arm"],r["scenario"]) for r in original]
    if len(keys)!=8 or set(keys)!=expected_original:
        raise ValueError("Incomplete original C2 score grid")
    expected_frozen={(a,s,c) for a in ("A","B") for s in noise.SEEDS for c in noise.CONDITIONS}
    keys=[(r["arm"],r["seed"],r["condition"]) for r in frozen]
    if len(keys)!=40 or set(keys)!=expected_frozen:
        raise ValueError("Incomplete frozen-noise score grid")
    old={r["scenario"]:r for r in original if r["arm"]=="A"}
    out=normalize_rows([old[s] for s in previous.SCENARIOS],"historical",None,previous.SCENARIOS,"A")
    for seed in noise.SEEDS:
        selected={r["condition"]:r for r in frozen if r["arm"]=="A" and r["seed"]==seed}
        out+=normalize_rows([selected[s] for s in noise.CONDITIONS],"seeded",seed,noise.CONDITIONS,"A")
    return out


def case_batches(halves):
    original=previous.evaluation_cases(halves)
    yield "historical",None,previous.SCENARIOS,original,None
    del original
    for seed in noise.SEEDS:
        cases,metadata=noise.paired_cases(halves,seed)
        mapped={old:cases[new] for old,new in zip(previous.SCENARIOS,noise.CONDITIONS)}
        yield "seeded",seed,noise.CONDITIONS,mapped,{"seed":seed,"halves":metadata}


def descriptor(family,seed,conditions,mapped,metadata):
    return {"family":family,"seed":seed,"perturbations":metadata,
            "observations":{condition:[{"period":d.period,"observations_sha256":noise.fingerprint(d),
                                        "truth_sha256":hashlib.sha256(y.tobytes()).hexdigest()}
                                       for d,y in mapped[old]]
                            for old,condition in zip(previous.SCENARIOS,conditions)}}


def replay_checks(rows,reference_lookup):
    checks=[]
    for row in rows:
        key=cell(row)
        if key not in reference_lookup:raise ValueError("Unexpected replay condition")
        target=reference_lookup[key]
        identity={"family":key[0],"seed":key[1],"condition":key[2]}
        for metric in noise.METRICS:
            checks.append({**identity,"check":metric,"actual":row[metric],
                           "recorded_token":str(target[metric]),
                           "passed":noise.recorded_precision_match(row[metric],target[metric])})
        checks.append({**identity,"check":"confusion","passed":row["confusion"]==target["confusion"]})
        if len(row["periods"])!=2 or len(target["periods"])!=2:
            raise ValueError("Expected two separately scored halves")
        for actual,saved in zip(row["periods"],target["periods"]):
            for field in ("prediction_sha256","availability_counts"):
                checks.append({**identity,"period":actual["period"],"check":field,
                               "passed":actual["period"]==saved["period"] and actual[field]==saved[field]})
    return checks


def reference_state(model,config):
    return (tuple(model.names),model.scaler.mean.tobytes(),model.scaler.scale.tobytes(),
            json.dumps(config,sort_keys=True))


def run_controlled(reference_model,reference_config,halves,reference,expected_perturbations,
                   pm,source_loader,folder,*,model_factory=RobustCRF,
                   on_replay=None,on_fit=None,on_variant=None):
    """Complete A replay gates every source load, fit and new-model evaluation."""
    noise.validate_config(reference_config,"A")
    before=reference_state(reference_model,reference_config)
    lookup={cell(r):r for r in reference}
    if len(reference)!=24 or set(lookup)!=set(CELLS):
        raise ValueError("Expected all 24 saved A conditions")
    perturbation_lookup={r["seed"]:r for r in expected_perturbations}
    if len(expected_perturbations)!=5 or set(perturbation_lookup)!=set(noise.SEEDS):
        raise ValueError("Expected the five saved perturbation manifests")
    reference_results,checks,descriptors=[],[],[]
    times={}
    start=time.perf_counter()
    for family,seed,conditions,mapped,metadata in case_batches(halves):
        rows=previous.evaluate_arm(reference_model,reference_config,mapped,"A",pm)
        rows=normalize_rows(rows,family,seed,conditions,"A")
        batch_checks=replay_checks(rows,lookup)
        if seed is not None:
            batch_checks.append({"family":family,"seed":seed,"check":"saved_noise_and_masks",
                                 "passed":metadata==perturbation_lookup[seed]})
        reference_results+=rows;checks+=batch_checks
        descriptors.append(descriptor(family,seed,conditions,mapped,metadata))
        if on_replay:on_replay(reference_results,checks,descriptors)
        if not all(check["passed"] for check in batch_checks):
            raise noise.ReplayMismatch("A replay mismatch; source loading and new fit were NOT started")
    if reference_state(reference_model,reference_config)!=before:
        raise RuntimeError("Frozen A changed during replay")
    times["A_replay_s"]=time.perf_counter()-start
    start=time.perf_counter()
    train,cal,records=source_loader()
    previous.validate_partitions(records,reference_config)
    times["load_source_partitions_s"]=time.perf_counter()-start
    model,fitted,fit_times=fit_variant(train,cal,reference_config,folder,model_factory=model_factory)
    times.update(fit_times)
    if model is reference_model:raise RuntimeError("Reference A cannot be the fitted variant")
    if on_fit:
        model,reloaded_config=on_fit(model,fitted,fit_times)
        if reloaded_config!=fitted:raise RuntimeError("Reload changed the new model configuration")
        fitted=reloaded_config
    require_schema(model)
    predictor=ProjectedPredictor(model)
    rows=list(reference_results)
    reference_by_cell={cell(r):r for r in reference_results}
    start=time.perf_counter()
    for index,(family,seed,conditions,mapped,metadata) in enumerate(case_batches(halves)):
        actual_descriptor=descriptor(family,seed,conditions,mapped,metadata)
        if actual_descriptor!=descriptors[index]:
            raise RuntimeError("Regenerated observations/masks/labels differ from A replay")
        batch=previous.evaluate_arm(predictor,fitted,mapped,"A",pm)
        batch=normalize_rows(batch,family,seed,conditions,VARIANT)
        for row in batch:
            baseline=reference_by_cell[cell(row)]
            if row["availability"]!=baseline["availability"] or any(
                a["availability_counts"]!=b["availability_counts"]
                for a,b in zip(row["periods"],baseline["periods"])):
                raise RuntimeError("Raw derivative availability must equal A")
            row["raw_availability_matches_A"]=True
        rows+=batch
        if on_variant:on_variant(rows)
    times["variant_evaluation_s"]=time.perf_counter()-start
    if reference_state(reference_model,reference_config)!=before:
        raise RuntimeError("Frozen A parameters changed")
    return rows,fitted,times,checks,descriptors


def comparisons(rows):
    keys=[(cell(r),r["variant"]) for r in rows]
    expected={(key,variant) for key in CELLS for variant in ("A",VARIANT)}
    if len(rows)!=48 or set(keys)!=expected:
        raise ValueError("Expected exactly 48 rows: 24 A and 24 ablated")
    lookup=dict(zip(keys,rows))
    paired,coverage=[],[]
    for family,seed,condition in CELLS:
        key=(family,seed,condition)
        a,n=lookup[key,"A"],lookup[key,VARIANT]
        common={"family":family,"seed":seed,"condition":condition}
        for metric in noise.METRICS:
            av,nv=float(a[metric]),float(n[metric])
            if any(not math.isfinite(v) or not 0<=v<=1 for v in (av,nv)):
                raise ValueError("Invalid F1")
            paired.append({**common,"metric":metric,"A":av,VARIANT:nv,
                           "variant_minus_A":nv-av,"variant_minus_A_pp":100*(nv-av)})
        for grid in ("25Hz","12.5Hz"):
            for measure in ("ball_velocity","player_velocity","ball_acceleration","home_velocity","away_velocity"):
                ca,cn=a["availability"][grid][measure],n["availability"][grid][measure]
                if ca!=cn:raise ValueError("Availability changed despite identical A derivatives")
                row={**common,"grid":grid,"measure":measure}
                for variant,values in (("A",ca),(VARIANT,cn)):
                    row.update({variant+"_"+k:v for k,v in values.items()})
                row["variant_minus_A_fraction_all_pp"]=0. if ca["fraction_all"] is not None else None
                row["variant_minus_A_fraction_observed_pp"]=0. if ca["fraction_observed"] is not None else None
                coverage.append(row)
    summary=[]
    for condition in noise.CONDITIONS:
        for metric in noise.METRICS:
            group=[r for r in paired if r["family"]=="seeded" and r["condition"]==condition and r["metric"]==metric]
            summary.append({"condition":condition,"metric":metric,
                            **{role:noise.descriptive([r[role] for r in group])
                               for role in ("A",VARIANT,"variant_minus_A")}})
    coverage_summary=[]
    for condition in noise.CONDITIONS:
        for grid in ("25Hz","12.5Hz"):
            for measure in ("ball_velocity","player_velocity","ball_acceleration","home_velocity","away_velocity"):
                group=[r for r in coverage if r["family"]=="seeded" and
                       (r["condition"],r["grid"],r["measure"])==(condition,grid,measure)]
                fields=("A_fraction_all",VARIANT+"_fraction_all","variant_minus_A_fraction_all_pp",
                        "A_fraction_observed",VARIANT+"_fraction_observed","variant_minus_A_fraction_observed_pp")
                coverage_summary.append({"condition":condition,"grid":grid,"measure":measure,
                                         **{f:noise.descriptive([r[f] for r in group]) for f in fields}})
    return paired,coverage,summary,coverage_summary
