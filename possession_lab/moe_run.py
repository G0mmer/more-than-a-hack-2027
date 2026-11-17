"""Fit source-only learned ensembles and evaluate complete held-out DFL matches."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch

from .baseline_suite import initialize,summarize,restart_recall,METRICS
from .ensemble import MotionBundle,decode
from .evaluation import official
from .moe import EXPERTS,prepare,fit
from .multimatch_run import MATCHES,RESERVED,SCENARIOS,partitions
from .probabilistic import probability_metrics
from .robust import mask_ball
from .robust_run import read_match,score
from .run import save_json,save_table
from .transformer_temporal import C3Bundle


def jitter(data,seed):
    rng=np.random.default_rng(seed)
    return replace(data,ball=data.ball+rng.normal(0,.3,data.ball.shape),
                   players=data.players+rng.normal(0,.3,data.players.shape))


def training_rows(calibration,tcn,a,d):
    es,xs,ys=[],[],[]
    for data,y,seed in calibration:
        # Fixed, historically used noise magnitude; no competition-derived fitting.
        cases=(data,mask_ball(data,.4,seed)[0],mask_ball(jitter(data,seed+5000),.4,seed)[0])
        for case in cases:
            e,x=prepare(case,tcn,a,d)
            es.append(e);xs.append(x);ys.append(y[::2])
    return np.concatenate(es),np.concatenate(xs),np.concatenate(ys)


def run(args):
    initialize(args.out,'learned-ensemble',dict(experts=list(EXPERTS),candidates=['tcn','weighted','moe'],
        matches=list(MATCHES),reserved_unopened=list(RESERVED),scenarios=SCENARIOS,gate_kind=args.gate_kind,
        gate_fit='Source calibration tails only; never the held-out match. Same tails already calibrate expert temperatures.',
        gate_input_conditions=['clean','missing40','iid_sigma_0.3m_plus_missing40'],
        gate_recipe={'epochs':100,'optimizer':'AdamW','lr':.03,'weight_decay':.01,'shrinkage':.001,'seed':7},
        decoder='Single shared TCN transition/prior/strength .25, once per half',
        probability_stage='calibrated native emissions mixed before decoding; no fitted post-chain temperature for mixtures',
        selection='Same clean, stress, dead-ball and restart guards as baseline_suite.compare',
        temporal_release=str(args.temporal_release.resolve()),tracking_run=str(args.tracking_run.resolve())))
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    datasets,files={},[]
    for match in MATCHES:
        print('READ EXTERNAL '+match,flush=True)
        datasets[match],audit,sources=read_match(args.dfl_root,match)
        save_json(args.out/(match+'_input_audit.json'),audit);files.extend(sources)
    save_json(args.out/'inputs.json',files)
    pm,rows=official(),[]
    for match in MATCHES:
        sources=[m for m in MATCHES if m!=match]
        folder=args.out/('leave_out_'+match);folder.mkdir()
        tcn=C3Bundle.load(args.temporal_release/'oof_models'/folder.name/'tcn')
        source=args.tracking_run/'folds'/folder.name
        a,d=MotionBundle(source/'A'),MotionBundle(source/'dead_specialist',True)
        if tcn.config['training_matches']!=sources or a.config['fold']['sources']!=sources:
            raise ValueError('Held-out source leakage')
        _,calibration,records=partitions(datasets,sources)
        save_json(folder/'gate_sources_before_fit.json',dict(training_matches=sources,partitions=records,
            expert_hashes={'A':a.hashes,'D':d.hashes},target_excluded=match))
        print('PREPARE GATE '+match,flush=True)
        tick=time.perf_counter()
        e,x,y=training_rows(calibration,tcn,a,d)
        np.savez_compressed(folder/'gate_training_inputs.npz',experts=e,quality=x,truth=y)
        gates={}
        kinds=(('weighted','dead_constant'),('moe','dead')) if args.gate_kind=='dead' else (('weighted','constant'),('moe','quality'))
        for name,kind in kinds:
            gate,history=fit(e,x,y,kind)
            gate.save(folder/name,dict(training_matches=sources,held_out=match,partitions=records,
                n_training_rows=len(y),conditions=['clean','missing40','noise30cm_missing40'],
                final_training_nll=history[-1],selection='Fixed epoch 100; held-out match excluded'))
            gates[name]=gate
        save_json(folder/'training.json',dict(seconds=time.perf_counter()-tick,rows=len(y)))
        del e,x,y
        for scenario,fraction,seed,long in SCENARIOS:
            cases=[(data if fraction is None else mask_ball(data,fraction,
                   seed+10*MATCHES.index(match)+i,long)[0],truth)
                   for i,(data,truth) in enumerate(datasets[match])]
            preds={name:[] for name in ('tcn','weighted','moe')}
            probs={name:[] for name in preds}
            weights={name:[] for name in gates}
            for data,_ in cases:
                e,x=prepare(data,tcn,a,d)
                emissions={'tcn':e[:,0]}
                for name,gate in gates.items():
                    emissions[name],w=gate.predict(e,x);weights[name].append(w.mean(0).tolist())
                for name,p in emissions.items():
                    q,y=decode(p,tcn.config)
                    probs[name].append(np.repeat(q,2,axis=0)[:len(data.ball)])
                    preds[name].append(np.repeat(y,2)[:len(data.ball)])
            for name in preds:
                row=score(pm,cases,preds[name])
                row.update(probability_metrics(np.concatenate(probs[name]),np.concatenate([y for _,y in cases]),np.concatenate(preds[name])))
                row.update(match=match,candidate=name,scenario=scenario,seed=seed,
                           restart_recall=restart_recall(pm,cases,preds[name]))
                rows.append(row)
                for (data,_),y in zip(cases,preds[name]):
                    np.savez_compressed(folder/f'{name}_{scenario}_{seed}_{data.period}.npz',frame=data.frame_ids,label=y)
            save_json(folder/f'weights_{scenario}_{seed}.json',weights)
            save_json(args.out/'scores_partial.json',rows)
            print(match,scenario,seed,{r['candidate']:round(r['spell_f1'],4) for r in rows[-3:]},flush=True)
    per,means,decision=summarize(rows,('tcn','weighted','moe'))
    save_json(args.out/'scores.json',rows);save_json(args.out/'means.json',means)
    save_table(args.out/'per_match.csv',per,['match','candidate','scenario',*METRICS])
    save_json(args.out/'decision.json',decision)
    print(json.dumps(decision,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dfl-root','temporal-release','tracking-run','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gate-kind',choices=('quality','dead'),default='quality')
    run(p.parse_args())


if __name__=='__main__':main()
