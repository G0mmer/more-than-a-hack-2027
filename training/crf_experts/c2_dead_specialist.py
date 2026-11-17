"""One C2 experiment: frozen full A plus a source-trained binary D specialist."""
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
import c2_no_team_motion as shared
import run_c2_dfl as previous
from possession_lab.data import sha256
from possession_lab.robust import items
from possession_lab.probabilistic import Standardizer, normalize, temperature, fit_temperature
from possession_lab.probabilistic_run import fusion

VARIANT="A_with_dead_specialist"
MODEL_DIR="dead_specialist"
NAMES=tuple(features.NAMES)
LABELS=("0","1")
MEANINGS={"0":"alive","1":"D"}
CELLS=shared.CELLS
ATOL=1e-12
RTOL=1e-10
CRF_PARAMS={"c1":0,"c2":.1,"max_iterations":80,
            "feature.possible_transitions":True,"feature.possible_states":True}
source_partitions=shared.source_partitions
cell=shared.cell
case_batches=shared.case_batches
descriptor=shared.descriptor
reference_rows=shared.reference_rows
normalize_rows=shared.normalize_rows
replay_checks=shared.replay_checks
reference_state=shared.reference_state
if len(NAMES)!=35:raise RuntimeError("Expected full original A schema")

class IdentityMismatch(RuntimeError):
    pass

def probabilities(values,columns):
    p=np.asarray(values,dtype=float)
    if (p.ndim!=2 or p.shape[1]!=columns or not len(p) or not np.isfinite(p).all()
            or np.any(p<0) or not np.allclose(p.sum(axis=1),1.,atol=ATOL,rtol=RTOL)):
        raise ValueError("Expected finite nonnegative normalized probabilities")
    return p

def calibrated(raw,value,columns):
    raw=probabilities(raw,columns)
    if not np.isfinite(value) or not .2<=value<=10:
        raise ValueError("Invalid saved emission temperature")
    return probabilities(temperature(raw,value),columns)

def compose(p,q):
    """Preserve conditional H/A before decoding; q replaces only D evidence.

    No extra clipping/temperature/normalization is applied. The existing
    temperature/normalize functions already protect tiny probabilities.
    Only an EXACTLY zero H+A denominator uses the symmetric h=0.5 convention;
    h is then undefined in A. No epsilon changes positive denominators.
    """
    p=probabilities(p,3)
    q=np.asarray(q,dtype=float)
    if q.shape!=(len(p),) or not np.isfinite(q).all() or np.any((q<0)|(q>1)):
        raise ValueError("Expected one finite D probability in [0,1] per frame")
    alive=p[:,0]+p[:,1]
    h=np.divide(p[:,0],alive,out=np.full(len(p),.5),where=alive>0)
    e=np.column_stack(((1-q)*h,(1-q)*(1-h),q))
    return probabilities(e,3)

def emissions(raw_A,raw_D,T_A,T_D):
    p=calibrated(raw_A,T_A,3)
    q=calibrated(raw_D,T_D,2)[:,1]
    return compose(p,q),p,q

def decode(e,reference_config):
    return fusion([probabilities(e,3)],[1.],np.asarray(reference_config["transition"]),
                  np.asarray(reference_config["prior"]),.25)[1]

def check_features(seq):
    if seq.x.ndim!=2 or seq.x.shape[1]!=35 or not len(seq.x) or not np.isfinite(seq.x).all():
        raise ValueError("Expected unchanged finite 35-feature sequence")

def binary_sequence(seq):
    check_features(seq)
    if seq.y is None or len(seq.y)!=len(seq.x) or not np.isin(seq.y,[0,1,2]).all():
        raise ValueError("Expected original complete H/A/D targets")
    if seq.full_y is None or not np.isin(seq.full_y,[0,1,2]).all():
        raise ValueError("Expected original full-frame targets")
    # No slicing or segmentation by the targets. Feature rows are unchanged.
    return replace(seq,y=(seq.y==2).astype(np.int8),full_y=(seq.full_y==2).astype(np.int8))

def training_sequences(train):
    return [binary_sequence(s) for s in previous.training_sequences(train,"A")]

def calibration_sequences(cal):
    return [binary_sequence(s) for s in previous.calibration_sequences(cal,"A")]

def check_labels(tagger):
    labels=list(tagger.labels())
    if len(labels)!=2 or set(labels)!=set(LABELS):
        raise ValueError("Binary CRF must contain exactly string labels 0=alive, 1=D")
    return labels

def check_scaler(scaler):
    if (scaler.mean.shape!=(35,) or scaler.scale.shape!=(35,)
            or scaler.mean.dtype.kind!="f" or scaler.scale.dtype.kind!="f"
            or not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.scale).all()
            or np.any(scaler.scale<.05)):
        raise ValueError("Expected a finite own 35-column scaler")

def compare_scalers(actual,reference):
    check_scaler(actual);check_scaler(reference)
    result={"atol":ATOL,"rtol":RTOL,
            "max_abs_mean_difference":float(np.max(np.abs(actual.mean-reference.mean))),
            "max_abs_scale_difference":float(np.max(np.abs(actual.scale-reference.scale))),
            "passed":bool(np.allclose(actual.mean,reference.mean,atol=ATOL,rtol=RTOL)
                          and np.allclose(actual.scale,reference.scale,atol=ATOL,rtol=RTOL))}
    if not result["passed"]:raise ValueError("Independent specialist scaler differs from A training scaler")
    return result

class BinaryCRF:
    def __init__(self,*,frozen=False):
        self.names=NAMES
        self.frozen=frozen
    def fit(self,seqs,folder,reference_scaler):
        if self.frozen:raise RuntimeError("Loaded specialist is frozen")
        if not seqs:raise ValueError("No source sequences")
        for s in seqs:
            check_features(s)
            if s.y is None or len(s.y)!=len(s.x) or not np.isin(s.y,[0,1]).all():
                raise ValueError("Expected binary targets")
        if set(np.concatenate([s.y for s in seqs]).tolist())!={0,1}:
            raise ValueError("Both alive and D are required in source training")
        self.scaler=Standardizer().fit(seqs)
        self.scaler_check=compare_scalers(self.scaler,reference_scaler)
        folder=Path(folder)
        folder.mkdir(parents=True,exist_ok=False)
        trainer=pycrfsuite.Trainer(verbose=False)
        for s in seqs:
            trainer.append(items(self.scaler.transform(s.x),self.names),[str(int(y)) for y in s.y])
        trainer.set_params(CRF_PARAMS.copy())
        trainer.train(str(folder/"model.crfsuite"))
        self.tagger=pycrfsuite.Tagger()
        if self.tagger.open(str(folder/"model.crfsuite")) is False:
            raise ValueError("Cannot open new binary CRF")
        self.native_labels=check_labels(self.tagger)
        np.savez(folder/"scaler.npz",mean=self.scaler.mean,scale=self.scaler.scale)
        return self
    def predict_proba(self,seq):
        if self.names!=NAMES:raise ValueError("Binary feature names/order changed")
        check_features(seq);check_scaler(self.scaler)
        check_labels(self.tagger)
        self.tagger.set(items(self.scaler.transform(seq.x),self.names))
        # Explicit string lookup: native tagger order may legitimately be 1,0.
        raw=np.asarray([[self.tagger.marginal(label,i) for label in LABELS]
                        for i in range(len(seq.x))],dtype=float)
        if not np.isfinite(raw).all() or np.any(raw<0) or np.any(raw.sum(axis=1)<=0):
            raise ValueError("Invalid binary CRF marginals")
        # Same normalization as A: tolerate CRFsuite sum roundoff on long chunks.
        return probabilities(normalize(raw),2)

def config_digest(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def validate_binary_config(config,reference_config):
    noise.validate_config(reference_config,"A")
    expected={"variant":VARIANT,"model_role":"binary_dead_specialist","names":list(NAMES),
              "labels":list(LABELS),"label_meanings":MEANINGS,"history":False,
              "input_hz":25,"stride":2,"context_samples":[-5,0,5],
              "velocity_backend":"five_frame","acceleration":"five_frame_velocity_difference",
              "training_variant":"motion_noise","training_matches":previous.SOURCES,
              "partitions":reference_config["partitions"],"crf":CRF_PARAMS,
              "reference_config_sha256":config_digest(reference_config)}
    if any(config.get(k)!=v for k,v in expected.items()):
        raise ValueError("Binary config/schema/label order/reference mismatch")
    if any(k in config for k in ("chain_temperature","emission_temperature","joint_temperature")):
        raise ValueError("The specialist has only its own binary_emission_temperature")
    td=config.get("binary_emission_temperature")
    if not isinstance(td,(int,float)) or not np.isfinite(td) or not .2<=td<=10:
        raise ValueError("Invalid binary temperature")

def fit_specialist(train,cal,reference_model,reference_config,folder):
    noise.validate_config(reference_config,"A")
    times={};tick=time.perf_counter()
    seqs=training_sequences(train)
    times["training_features_s"]=time.perf_counter()-tick
    tick=time.perf_counter()
    model=BinaryCRF().fit(seqs,folder,reference_model.scaler)
    times["binary_crf_fit_s"]=time.perf_counter()-tick
    del seqs
    tick=time.perf_counter();seqs=calibration_sequences(cal)
    times["calibration_features_s"]=time.perf_counter()-tick
    tick=time.perf_counter()
    raw=[model.predict_proba(s) for s in seqs]
    td=fit_temperature(np.concatenate(raw),np.concatenate([s.y for s in seqs]))
    times["binary_calibration_predict_and_fit_s"]=time.perf_counter()-tick
    config={"variant":VARIANT,"model_role":"binary_dead_specialist","names":list(NAMES),
            "labels":list(LABELS),"label_meanings":MEANINGS.copy(),"history":False,
            "input_hz":25,"stride":2,"context_samples":[-5,0,5],
            "velocity_backend":"five_frame","acceleration":"five_frame_velocity_difference",
            "training_variant":"motion_noise","training_matches":list(previous.SOURCES),
            "partitions":copy.deepcopy(reference_config["partitions"]),"crf":CRF_PARAMS.copy(),
            "reference_config_sha256":config_digest(reference_config),
            "binary_emission_temperature":float(td),"tagger_native_labels":model.native_labels,
            "scaler_matches_A":model.scaler_check}
    validate_binary_config(config,reference_config)
    return model,config,times

def load_specialist(folder,hashes,reference_model,reference_config,*,tagger_factory=None):
    folder=Path(folder)
    if set(hashes)!=set(noise.MODEL_FILES):raise ValueError("Expected all three binary artifact hashes")
    for name in noise.MODEL_FILES:
        if sha256(folder/name)!=hashes[name]:raise ValueError("Binary artifact hash mismatch: "+name)
    config=json.loads((folder/"config.json").read_text(encoding="utf-8-sig"))
    validate_binary_config(config,reference_config)
    model=BinaryCRF(frozen=True);model.scaler=Standardizer()
    with np.load(folder/"scaler.npz",allow_pickle=False) as saved:
        if set(saved.files)!={"mean","scale"}:raise ValueError("Unexpected scaler fields")
        model.scaler.mean,model.scaler.scale=saved["mean"].copy(),saved["scale"].copy()
    model.scaler_check=compare_scalers(model.scaler,reference_model.scaler)
    model.scaler.mean.setflags(write=False);model.scaler.scale.setflags(write=False)
    model.tagger=(tagger_factory or pycrfsuite.Tagger)()
    if model.tagger.open(str(folder/"model.crfsuite")) is False:raise ValueError("Cannot open binary CRF")
    model.native_labels=check_labels(model.tagger)
    return model,config

def dead_diagnostics(pm,halves,predictions):
    """Existing frozen_eval.diagnose boundary recipe, pooled across separate halves.

    Timing errors are prediction-minus-truth boundary estimates, not online latency.
    """
    totals={kind:{"tp":0,"fp":0,"fn":0,"signed_errors_s":[]} for kind in ("stop","restart")}
    false_dead=missed_dead=0
    if len(halves)!=len(predictions):raise ValueError("Expected one prediction per half")
    for (_,truth),prediction in zip(halves,predictions):
        if len(truth)!=len(prediction) or not np.isin(prediction,[0,1,2]).all():
            raise ValueError("Expected complete 3-state predictions")
        false_dead+=int(np.count_nonzero((truth<2)&(truth>=0)&(prediction==2)))
        missed_dead+=int(np.count_nonzero((truth==2)&(prediction<2)))
        runs=pm.valid_runs(truth)
        filled=pm.fill_abstentions(prediction,runs)
        for run in runs:
            gt=pm.transitions(truth,[run]);pr=pm.transitions(filled,[run])
            for kind in totals:
                def select(values):
                    if kind=="stop":return [(i,0,1) for i,a,b in values if a<2 and b==2]
                    return [(i,1,0) for i,a,b in values if a==2 and b<2]
                pairs,missed,spurious=pm.match_transitions(select(gt),select(pr),25,"pair")
                t=totals[kind];t["tp"]+=len(pairs);t["fp"]+=len(spurious);t["fn"]+=len(missed)
                t["signed_errors_s"].extend((p[0]-g[0])/25 for g,p in pairs)
    for t in totals.values():
        tp,fp,fn=t["tp"],t["fp"],t["fn"]
        t.update(precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
                 f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        signed=t["signed_errors_s"];absolute=list(map(abs,signed))
        t.update(absolute_errors_s=absolute,mean_signed_error_s=float(np.mean(signed)) if signed else None,
                 median_signed_error_s=float(np.median(signed)) if signed else None,
                 mean_absolute_error_s=float(np.mean(absolute)) if signed else None,
                 median_absolute_error_s=float(np.median(absolute)) if signed else None)
    return {"false_dead_frames":false_dead,"missed_dead_frames":missed_dead,
            "false_dead_seconds":false_dead/25,"missed_dead_seconds":missed_dead/25,
            "tolerance_frames":25,"boundaries":totals}

def evaluate_batch(reference_model,reference_config,cases,pm,*,binary_model=None,
                   binary_config=None,identity=False,artifact_callback=None):
    """Local evaluator; composed calibrated emissions never enter evaluate_arm."""
    if binary_model is not None:validate_binary_config(binary_config,reference_config)
    if binary_model is not None and identity:raise ValueError("Identity is a pre-fit A control")
    rows,identity_checks=[],[]
    for scenario in previous.SCENARIOS:
        start=time.perf_counter()
        scores,counts_all,periods,predictions=[],[],[],[]
        timing={k:0. for k in ("features_and_availability_s","predict_decode_s","identity_control_s",
                               "scoring_s","dead_diagnostics_s")}
        for data,truth in cases[scenario]:
            tick=time.perf_counter()
            seq,counts=features.make_sequence(data,None,"A",diagnostics=True)
            timing["features_and_availability_s"]+=time.perf_counter()-tick
            tick=time.perf_counter()
            raw_A=reference_model.predict(seq)[0]
            if binary_model is None:
                p=calibrated(raw_A,reference_config["emission_temperature"],3)
                q=p[:,2];e=p
            else:
                e,p,q=emissions(raw_A,binary_model.predict_proba(seq),
                                reference_config["emission_temperature"],
                                binary_config["binary_emission_temperature"])
            path=decode(e,reference_config)
            prediction=np.repeat(path,previous.STRIDE)[:len(truth)]
            timing["predict_decode_s"]+=time.perf_counter()-tick
            if identity:
                tick=time.perf_counter()
                reconstructed=compose(p,p[:,2])
                restored=decode(reconstructed,reference_config)
                delta=float(np.max(np.abs(reconstructed-p)))
                equal=bool(np.allclose(reconstructed,p,atol=ATOL,rtol=RTOL))
                mismatch=int(np.count_nonzero(restored!=path))
                identity_checks.append({"scenario":scenario,"period":data.period,
                    "max_abs_emission_error":delta,"atol":ATOL,"rtol":RTOL,
                    "emissions_match":equal,"path_mismatches":mismatch,
                    "passed":equal and mismatch==0})
                timing["identity_control_s"]+=time.perf_counter()-tick
            tick=time.perf_counter()
            score=previous.evaluate_period(pm,truth,prediction)
            timing["scoring_s"]+=time.perf_counter()-tick
            scores.append(score);counts_all.append(counts);predictions.append(prediction)
            alive=p[:,0]+p[:,1];mass=e[:,0]+e[:,1]
            valid=(alive>0)&(mass>0)
            ratio_error=float(np.max(np.abs(e[valid,0]/mass[valid]-p[valid,0]/alive[valid]))) if valid.any() else None
            period={"period":data.period,"availability_counts":counts,"scoring_counts":score,
                    "prediction_sha256":hashlib.sha256(prediction.tobytes()).hexdigest(),
                    "probability_diagnostics":{"max_sum_error":float(np.max(np.abs(e.sum(axis=1)-1))),
                        "max_conditional_H_ratio_error":ratio_error,
                        "D_min":float(q.min()),"D_max":float(q.max()),
                        "zero_A_alive_mass_frames":int(np.count_nonzero(alive==0))}}
            if artifact_callback:
                artifact=artifact_callback(scenario,data,{"frame_ids_25Hz":data.frame_ids,
                    "prediction_25Hz":prediction,"p_A_12_5Hz":p,"q_D_12_5Hz":q,"emissions_12_5Hz":e})
                if artifact is not None:period["probability_artifact"]=artifact
            periods.append(period)
        tick=time.perf_counter()
        diagnostics=dead_diagnostics(pm,cases[scenario],predictions)
        timing["dead_diagnostics_s"]+=time.perf_counter()-tick
        arm="A" if binary_model is None else VARIANT
        rows.append({"arm":arm,"variant":arm,"scenario":scenario,**previous.aggregate(pm,scores),
                     "availability":previous.pooled_availability(counts_all),"periods":periods,
                     "dead_diagnostics":diagnostics,"timing":timing,"elapsed_s":time.perf_counter()-start})
    return rows,identity_checks

def run_controlled(reference_model,reference_config,halves,reference,expected_perturbations,
                   pm,source_loader,folder,*,on_replay=None,on_fit=None,on_variant=None,
                   artifact_callback=None):
    noise.validate_config(reference_config,"A")
    before=reference_state(reference_model,reference_config)
    lookup={cell(r):r for r in reference}
    if len(reference)!=24 or set(lookup)!=set(CELLS):raise ValueError("Expected all 24 A conditions")
    perturbations={r["seed"]:r for r in expected_perturbations}
    if len(expected_perturbations)!=5 or set(perturbations)!=set(noise.SEEDS):
        raise ValueError("Expected all five perturbation manifests")
    results,checks,descriptors,identities=[],[],[],[]
    def callback(variant,family,seed,conditions):
        if artifact_callback is None:return None
        mapping=dict(zip(previous.SCENARIOS,conditions))
        return lambda scenario,data,arrays:artifact_callback(
            variant,family,seed,mapping[scenario],data,arrays)
    tick=time.perf_counter()
    for family,seed,conditions,mapped,metadata in case_batches(halves):
        batch,identity_batch=evaluate_batch(reference_model,reference_config,mapped,pm,identity=True,
            artifact_callback=callback("A",family,seed,conditions))
        batch=normalize_rows(batch,family,seed,conditions,"A")
        batch_checks=replay_checks(batch,lookup)
        if seed is not None:
            batch_checks.append({"family":family,"seed":seed,"check":"saved_noise_and_masks",
                                 "passed":metadata==perturbations[seed]})
        condition_map=dict(zip(previous.SCENARIOS,conditions))
        for item in identity_batch:
            item.update(family=family,seed=seed,condition=condition_map[item.pop("scenario")])
        results+=batch;checks+=batch_checks;identities+=identity_batch
        descriptors.append(descriptor(family,seed,conditions,mapped,metadata))
        if on_replay:on_replay(results,checks,descriptors,identities)
        if not all(c["passed"] for c in batch_checks):
            raise noise.ReplayMismatch("A replay mismatch; sources and specialist fit were NOT started")
        if not all(c["passed"] for c in identity_batch):
            raise IdentityMismatch("A identity mismatch; sources and specialist fit were NOT started")
    if len(checks)!=197 or len(identities)!=48:
        raise RuntimeError("Incomplete 197 replay checks / 48 half-sequence identity controls")
    if reference_state(reference_model,reference_config)!=before:raise RuntimeError("Frozen A changed")
    times={"A_replay_and_identity_s":time.perf_counter()-tick}
    tick=time.perf_counter();train,cal,records=source_loader()
    previous.validate_partitions(records,reference_config)
    times["load_source_partitions_s"]=time.perf_counter()-tick
    model,config,fit_times=fit_specialist(train,cal,reference_model,reference_config,folder)
    times.update(fit_times)
    if on_fit:
        model,reloaded=on_fit(model,config,fit_times)
        if reloaded!=config:raise RuntimeError("Specialist config changed on reload")
        config=reloaded
    baseline={cell(r):r for r in results}
    tick=time.perf_counter()
    for index,(family,seed,conditions,mapped,metadata) in enumerate(case_batches(halves)):
        if descriptor(family,seed,conditions,mapped,metadata)!=descriptors[index]:
            raise RuntimeError("Regenerated observations/masks/labels differ from A replay")
        batch,_=evaluate_batch(reference_model,reference_config,mapped,pm,binary_model=model,
            binary_config=config,artifact_callback=callback(VARIANT,family,seed,conditions))
        batch=normalize_rows(batch,family,seed,conditions,VARIANT)
        for row in batch:
            a=baseline[cell(row)]
            if row["availability"]!=a["availability"] or any(
                    p["availability_counts"]!=q["availability_counts"] for p,q in zip(row["periods"],a["periods"])):
                raise RuntimeError("Raw derivative availability must equal A")
            row["raw_availability_matches_A"]=True
        results+=batch
        if on_variant:on_variant(results)
    times["candidate_evaluation_s"]=time.perf_counter()-tick
    if reference_state(reference_model,reference_config)!=before:raise RuntimeError("Frozen A changed")
    return results,config,times,checks,descriptors,identities

def diagnostic_comparisons(rows):
    if len(rows)!=48:raise ValueError("Expected all 48 rows")
    lookup={(cell(r),r["variant"]):r for r in rows}
    if set(lookup)!={(c,v) for c in CELLS for v in ("A",VARIANT)}:raise ValueError("Incomplete grid")
    paired=[]
    fields=("tp","fp","fn","precision","recall","f1","mean_signed_error_s",
            "median_signed_error_s","mean_absolute_error_s","median_absolute_error_s")
    for family,seed,condition in CELLS:
        key=(family,seed,condition)
        a,n=(lookup[key,v]["dead_diagnostics"] for v in ("A",VARIANT))
        measures=[(metric,a[metric],n[metric]) for metric in ("false_dead_seconds","missed_dead_seconds")]
        measures += [(kind+"_"+field,a["boundaries"][kind][field],n["boundaries"][kind][field])
                     for kind in ("stop","restart") for field in fields]
        for metric,av,nv in measures:
            paired.append({"family":family,"seed":seed,"condition":condition,"metric":metric,
                           "A":av,VARIANT:nv,"variant_minus_A":None if av is None or nv is None else nv-av})
    summary=[]
    for condition in noise.CONDITIONS:
        for metric in sorted({r["metric"] for r in paired}):
            group=[r for r in paired if r["family"]=="seeded" and r["condition"]==condition and r["metric"]==metric]
            summary.append({"condition":condition,"metric":metric,
                **{role:noise.descriptive([r[role] for r in group]) for role in ("A",VARIANT,"variant_minus_A")}})
    return paired,summary


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
