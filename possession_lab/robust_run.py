"""Public-only frozen robustness ablation, then a predeclared third match."""
from .external_paths import allowed_output
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
from .data import load_dfl, sha256
from .models import features, baseline, rules, RuleConfig
from .probabilistic import split_ranges, transition_parameters, temperature, fit_temperature, expand, probability_metrics, STRIDE
from .probabilistic_run import fusion
from .robust import VARIANTS, RobustCRF, Bundle, crop, mask_ball, sequence
from .evaluation import official, evaluate_period, aggregate
from .run import save_json, save_table

def score(pm, halves, preds):
    return aggregate(pm,[evaluate_period(pm,y,p) for (_,y),p in zip(halves,preds)])

def read_match(root, match):
    directory=root/match
    meta=list(directory.glob("*matchinformation*"+match+".xml"))
    positions=list(directory.glob("*positions_raw_observed*"+match+".xml"))
    if len(meta)!=1 or len(positions)!=1:
        raise ValueError("Missing public DFL files: "+match)
    halves,audit=load_dfl(meta[0],positions[0])
    if audit["match_id"]!="DFL-MAT-"+match or any(not np.isin(y,[0,1,2]).all() for d,y in halves):
        raise ValueError("Unexpected public match or incomplete reference")
    return halves,audit,[{"path":str(p),"sha256":sha256(p)} for p in (meta[0],positions[0])]

def calibration_cases(halves):
    return halves+[(mask_ball(d,.4,103+i)[0],y) for i,(d,y) in enumerate(halves)]

def train_source(source, halves, old, out, pm):
    dest=out/source
    dest.mkdir()
    train,cal=[],[]
    for data,y in halves:
        ranges=split_ranges(len(y))
        a,b=ranges["refit"]
        train.append((crop(data,a,b),y[a:b]))
        a,b=ranges["calibrate"]
        cal.append((crop(data,a,b),y[a:b]))
    cleanseq=[sequence(d,False,y) for d,y in train]
    transition,prior=transition_parameters(cleanseq)
    cases=calibration_cases(cal)
    truth=np.concatenate([y for d,y in cases])
    records,bundles=[],{}
    for name,(history,aug) in VARIANTS.items():
        folder=dest/name
        if name=="control":
            folder.mkdir()
            shutil.copyfile(old/"models"/"crf_context_c01_final.crfsuite",folder/"model.crfsuite")
            shutil.copyfile(old/"models"/"crf_context_c01_final.scaler.npz",folder/"scaler.npz")
            saved=json.loads((old/"results.json").read_text(encoding="utf-8"))
            fb=json.loads((old/"fusion_artifact.json").read_text(encoding="utf-8"))
            if fb["chosen_models"]["crf"]!="crf_context_c01" or saved["fusion"]["ensemble_chain025"]["weights"]!=[0.,0.,1.]:
                raise ValueError("Control must match the previously frozen CRF-only ensemble")
            temp=next(r["temperature"] for r in saved["rows"] if r["model"]=="crf_context_c01")
            config={"history":False,"stride":STRIDE,"emission_temperature":temp,
                    "chain_temperature":saved["fusion"]["ensemble_chain025"]["temperature"],
                    "transition":fb["transition"],"prior":fb["prior"]}
            model=RobustCRF.load(folder,config)
        else:
            training=list(train)
            if aug:
                training += [(mask_ball(d,.4,7+i)[0],y) for i,(d,y) in enumerate(train)]
            seqs=[sequence(d,history,y) for d,y in training]
            print(source+" training "+name,flush=True)
            model=RobustCRF(history).fit(seqs,folder)
            cal_seqs=[sequence(d,history,y) for d,y in cases]
            raw=[model.predict(s)[0] for s in cal_seqs]
            ysub=np.concatenate([s.y for s in cal_seqs])
            temp=fit_temperature(np.concatenate(raw),ysub)
            chained=[fusion([temperature(p,temp)],[1.],transition,prior,.25) for p in raw]
            chain_temp=fit_temperature(np.concatenate([p for p,y in chained]),ysub)
            config={"history":history,"stride":STRIDE,"emission_temperature":temp,"chain_temperature":chain_temp,
                    "transition":transition.tolist(),"prior":prior.tolist()}
        config.update(variant=name,source=source,augmentation=aug,c2=.1,context=5,chain_strength=.25,
                      target_fit=False,calibration="source last 20%, clean + missing40, fixed seed103/104")
        save_json(folder/"bundle.json",config)
        bundle=Bundle(model,config)
        pred=[bundle.predict(d) for d,y in cases]
        clean=score(pm,cal,[y for p,y in pred[:len(cal)]])["spell_f1"]
        missing=score(pm,cases[len(cal):],[y for p,y in pred[len(cal):]])["spell_f1"]
        metric=probability_metrics(np.concatenate([p for p,y in pred]),truth,np.concatenate([y for p,y in pred]))
        records.append({"variant":name,"clean_spell":clean,"missing40_spell":missing,
                        "selection_score":(clean+missing)/2,"calibration_nll":metric["nll"]})
        bundles[name]=bundle
        print(source+" calibrated "+name,flush=True)
    chosen=max(records,key=lambda r:r["selection_score"])["variant"]
    save_json(dest/"selection_before_target.json",{"selected":chosen,"source_metrics":records,
        "selection_rule":"equal-weight mean clean/missing40 spellF1 on SOURCE calibration only",
        "limitations":"calibration block also used for selection; target remains separate"})
    return bundles,chosen

def evaluate_target(source,target,halves,bundles,chosen,out,pm,third=False):
    scenarios=[("clean",0,None,False)]
    for fraction in (.2,.4):
        for seed in ((1000,) if third else (1000,1001,1002)):
            scenarios.append((f"missing_{fraction}",seed,fraction,False))
    if not third:
        scenarios.append(("long_missing_0.4",1000,.4,True))
    rows=[]
    dest=out/(source+"_to_"+target)
    dest.mkdir()
    for scenario,seed,fraction,long in scenarios:
        cases=[(d if fraction is None else mask_ball(d,fraction,seed+i,long)[0],y) for i,(d,y) in enumerate(halves)]
        for name,bundle in bundles.items():
            pred=[bundle.predict(d) for d,y in cases]
            metric=score(pm,cases,[y for p,y in pred])
            prob=probability_metrics(np.concatenate([p for p,y in pred]),np.concatenate([y for d,y in cases]),
                                     np.concatenate([y for p,y in pred]))
            metric.update(source=source,target=target,variant=name,scenario=scenario,seed=seed,
                          source_selected=name==chosen,nll=prob["nll"],ece=prob["ece"],brier=prob["brier"])
            rows.append(metric)
            if scenario=="clean":
                for (d,y),(p,prediction) in zip(cases,pred):
                    np.savez_compressed(dest/(name+"_"+d.period+".npz"),p=p.astype(np.float32),label=prediction,frame=d.frame_ids)
            print(f"{source}->{target} {scenario}/{seed} {name}: {metric['spell_f1']:.4f}",flush=True)
        for name in ("B50","rules_V0"):
            preds=[baseline(features(d),50) if name=="B50" else rules(features(d),RuleConfig(.2,.08,.4,True)) for d,y in cases]
            metric=score(pm,cases,preds)
            metric.update(source=source,target=target,variant=name,scenario=scenario,seed=seed,source_selected=False)
            rows.append(metric)
        save_json(dest/"scores.json",rows)
    save_table(dest/"scores.csv",rows,["source","target","variant","scenario","seed","source_selected","spell_f1",
                                     "macro_f1","f1_D","gt_spells","pred_spells","nll","ece","brier"])
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfl-root",required=True,type=Path)
    parser.add_argument("--previous-run",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--third-match",choices=["J03WPY"],default="J03WPY")
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    if args.out.exists() or (not allowed_output(args.out, repo)):
        parser.error("Output must be new and under data/external/runs or outside repo")
    oldmanifest=json.loads((args.previous_run/"manifest.json").read_text(encoding="utf-8"))
    hashes={k.replace("\\","/"):v for k,v in oldmanifest["code_sha256"].items()}
    for name in ("data.py","models.py","probabilistic.py","probabilistic_run.py","evaluation.py"):
        if hashes["possession_lab/"+name]!=sha256(repo/"possession_lab"/name):
            raise ValueError("Previous model code changed")
    if oldmanifest["scorer_sha256"]!=sha256(repo/"evaluation/possession_metrics.py"):
        raise ValueError("Scorer changed")
    args.out.mkdir(parents=True)
    protocol={"variants":VARIANTS,"sources":["J03WMX","J03WN1"],"third_match":args.third_match,
       "train":"source first 80%, 5s purge; clean plus independent masked40 copy for augmented variants",
       "calibration":"source final 20%, 5s purge, clean+masked40; new variants only",
       "hyperparameters":{"c2":.1,"context_samples":5,"chain_strength":.25,"stride":STRIDE},
       "seeds":{"train":[7,8],"calibrate":[103,104],"test":[1000,1001,1002]},
       "test_masks":"short blocks 0.52-2s, long blocks 2-8s, independent of labels",
       "limitations":["Synthetic missingness is not actual Metrica missingness.",
                      "Both initial DFL matches are development data.",
                      "History ablation uses source-calibration including missing cases; control preserves old calibration.",
                      "Source selection/calibration share a block; no target selection.",
                      "Competition input/reference never opened; held-out J03WQQ/J03WR9 not opened."]}
    save_json(args.out/"protocol_before_run.json",protocol)
    datasets,audits,sources={},{},[]
    for match in ("J03WMX","J03WN1"):
        datasets[match],audits[match],files=read_match(args.dfl_root,match)
        sources+=files
    pm=official(repo)
    trained,chosen={},{}
    # Freeze both candidates before any new third-match read.
    for source,target in (("J03WMX","J03WN1"),("J03WN1","J03WMX")):
        trained[source],chosen[source]=train_source(source,datasets[source],
            args.previous_run/(source+"_to_"+target),args.out,pm)
    save_json(args.out/"frozen_before_third_match.json",{"selected":chosen,
        "bundle_sha256":{str(p.relative_to(args.out)):sha256(p) for p in args.out.glob("J03*/**/bundle.json")},
        "utc":datetime.now(timezone.utc).isoformat()})
    rows=[]
    for source,target in (("J03WMX","J03WN1"),("J03WN1","J03WMX")):
        rows+=evaluate_target(source,target,datasets[target],trained[source],chosen[source],args.out,pm)
    third,audits[args.third_match],files=read_match(args.dfl_root,args.third_match)
    sources+=files
    for source in trained:
        rows+=evaluate_target(source,args.third_match,third,trained[source],chosen[source],args.out,pm,third=True)
    save_json(args.out/"summary.json",rows)
    save_json(args.out/"audits.json",audits)
    save_json(args.out/"manifest.json",{"sources":sources,"selected":chosen,"utc":datetime.now(timezone.utc).isoformat(),
        "code_sha256":{str(p.relative_to(repo)):sha256(p) for p in (repo/"possession_lab").glob("*.py")},
        "scorer_sha256":sha256(repo/"evaluation/possession_metrics.py")})
    lines=["# Public DFL robustness ablation","",
       "All new configurations fixed before run. Source-selected model is marked *. No competition data used.","",
       "| Source → target | Scenario | Model | Mean spellF1 | Mean macroF1 | Seeds |",
       "|---|---|---|---:|---:|---:|"]
    groups={}
    for r in rows:
        groups.setdefault((r["source"],r["target"],r["scenario"],r["variant"]),[]).append(r)
    for (s,t,scenario,name),values in groups.items():
        mark=" *" if name==chosen[s] else ""
        lines.append(f"| {s} → {t} | {scenario} | {name}{mark} | {np.mean([v['spell_f1'] for v in values]):.4f} | {np.mean([v['macro_f1'] for v in values]):.4f} | {len(values)} |")
    (args.out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("COMPLETE "+str(args.out),flush=True)

if __name__=="__main__": main()
