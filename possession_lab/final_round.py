"""Portable external-data screen, confirmation and frozen refits for the final round."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .baseline_suite import environment, summarize
from .be_ha_gate import visibility_gate, mix_home_away
from .data import Tracking, sha256
from .ensemble import MotionBundle, decode
from .evaluation import official
from .moe import Gate, prepare as gate_inputs, fit as fit_gate
from .moe_run import training_rows
from .motion_tcn_metrics import diagnostics
from .multimatch_run import MATCHES, SCENARIOS, partitions
from .probabilistic import probability_metrics, temperature
from .robust import mask_ball
from .run import save_json
from .stress import perturb
from .symmetry import SymmetryAdapter
from .symmetry_train import RawSymmetryEpochs
from .team_temporal import TeamConfig, TeamTemporalBundle, fit, prepare

RECIPES = {
    'geometry': dict(),
    'motion': dict(feature_set='motion'),
    'masked': dict(feature_set='motion_masked'),
    'masked_aug': dict(feature_set='motion_masked', augment=True),
    'masked32': dict(feature_set='motion_masked', context_seconds=32),
    'masked_wide': dict(feature_set='motion_masked', hidden=44),
    'deepsets': dict(representation='deepsets'),
    'graph': dict(representation='graph'),
}
ARMS = ('plain', 'moe_d', 'moe_d_crf', 'quality', 'p1')
SEEDS = (7, 17, 27)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(root):
    manifest = read_json(root/'manifest.json')
    if manifest['schema'] != 'external-final-round-input-v1':
        raise ValueError('Wrong input package')
    for name, digest in manifest['sha256'].items():
        if sha256(root/name) != digest:
            raise ValueError('Changed input '+name)
    return manifest


def load_tracking(path):
    with np.load(path, allow_pickle=False) as z:
        d = Tracking(str(z['period']), z['frame_ids'], z['ball'], z['players'],
                     z['teams'], tuple(z['pitch']), float(z['fps']))
        y = z['truth'].copy() if 'truth' in z else None
    return d, y


def datasets(root):
    return {m: [load_tracking(root/'dfl'/f'{m}_{half}.npz')
                for half in ('firstHalf','secondHalf')] for m in MATCHES}


def recipe(name, seed, epochs=8):
    spec = dict(RECIPES[name]); spec.pop('augment', None)
    return TeamConfig(**spec, seed=seed, epochs=epochs)


def views(rows, config):
    def make(d, y):
        return prepare(d, config.representation != 'geometry', y, config.feature_set)
    clean = [make(d,y) for d,y,_ in rows]
    return clean + [make(mask_ball(d,.4,s)[0],y) for d,y,s in rows], clean


def get_model(folder, name, seed, train, cal, records, sources, device, epochs):
    cfg = recipe(name,seed,epochs)
    if (folder/'complete.json').exists():
        bundle = TeamTemporalBundle.load(folder,device)
        if (bundle.config['model'] != asdict(cfg) or bundle.config['training_matches'] != sources
                or bundle.config['partitions'] != records):
            raise ValueError('Resume model mismatch')
        return bundle
    if folder.exists():
        # Preserve interrupted writes for inspection; completed models are never moved.
        archive=folder.with_name(folder.name+'.interrupted-'+str(time.time_ns()))
        if archive.resolve().parent!=folder.resolve().parent:raise ValueError('Unsafe archive target')
        folder.rename(archive)
    training,clean = views(train,cfg); calibration,_ = views(cal,cfg)
    aug = RawSymmetryEpochs(train,seed,cfg.feature_set,cfg.representation!='geometry') if RECIPES[name].get('augment') else None
    model = fit(training,calibration,clean,sources,records,folder,cfg,device,epoch_training=aug)
    save_json(folder/'augmentation.json', [] if aug is None else aug.history)
    return model


def motion_pair(root, held):
    folder = root/'experts'/('final' if held is None else 'leave_out_'+held)
    a,d = MotionBundle(folder/'A'), MotionBundle(folder/'dead_specialist',True)
    sources = list(MATCHES) if held is None else [m for m in MATCHES if m!=held]
    for b in (a,d):
        if b.config['fold']['sources'] != sources:
            raise ValueError('Wrong CRF source matches')
    return a,d


def get_gates(folder, tcn, a, d, cal, sources):
    paths = [folder/'gate_dead',folder/'gate_quality']
    if all((p/'complete.json').exists() for p in paths):
        pairs=[Gate.load(p) for p in paths]
        if any(meta['training_matches']!=sources for _,meta in pairs):
            raise ValueError('Wrong gate source matches')
        return [g for g,_ in pairs]
    e,x,y=training_rows(cal,tcn,a,d)
    gates=[]
    for kind,path in zip(('dead','quality'),paths):
        if (path/'complete.json').exists():
            g,meta=Gate.load(path)
            if meta['training_matches']!=sources: raise ValueError('Gate sources changed')
        else:
            if path.exists():
                archive=path.with_name(path.name+'.interrupted-'+str(time.time_ns()))
                if archive.resolve().parent!=path.resolve().parent:raise ValueError('Unsafe archive target')
                path.rename(archive)
            g,history=fit_gate(e,x,y,kind)
            g.save(path,dict(training_matches=sources,rows=len(y),source_loss=history[-1],
                            calibration='same source tails as expert temperature; not independent'))
        gates.append(g)
    return gates


def infer(tcn,a,d,gates,data):
    e,x=gate_inputs(data,tcn,a,d)
    pdead,weights=gates[0].predict(e,x)
    pquality,qweights=gates[1].predict(e,x)
    eligible,_,_=visibility_gate(data)
    mixed,_=mix_home_away(pdead,e[:,1],eligible)
    adapter=SymmetryAdapter(tcn)
    return {'plain':(e[:,0],tcn.config),'moe_d':(pdead,tcn.config),
            'moe_d_crf':(mixed,tcn.config),'quality':(pquality,tcn.config),
            'p1':(adapter.emissions(data),adapter.config)}, {'moe_d':weights,'quality':qweights}


def score_cases(tcn,a,d,gates,cases,originals,folder,meta):
    pm=official(); preds={k:[] for k in ARMS}; probs={k:[] for k in ARMS}
    hashes={}; tick=time.perf_counter()
    for data,_ in cases:
        arms,weights=infer(tcn,a,d,gates,data)
        for name,(emission,config) in arms.items():
            q,y=decode(emission,config)
            q=np.repeat(q,2,axis=0)[:len(data.ball)]; y=np.repeat(y,2)[:len(data.ball)]
            preds[name].append(y); probs[name].append(q)
            path=folder/f'{name}_{data.period}.npz'
            # References never enter prediction exports.
            extra={'gate_weights':weights[name].astype(np.float32)} if name in weights else {}
            np.savez_compressed(path,frame=data.frame_ids[::2],label_frame=data.frame_ids,
                emission=emission.astype(np.float32),label=y.astype(np.int8),classes=np.array(['H','A','D']),
                probability_stage='calibrated native emission before shared decoder; final labels at 25Hz',
                match=meta['match'],half=data.period,ball_seen=np.isfinite(data.ball).all(1),
                **extra)
            hashes[path.name]=sha256(path)
    rows=[]
    for arm in ARMS:
        row=diagnostics(pm,originals,preds[arm])
        row.update(probability_metrics(np.concatenate(probs[arm]),np.concatenate([y for _,y in cases]),np.concatenate(preds[arm])))
        rows.append(dict(**row,**meta,arm=arm,candidate=meta['recipe']+'__'+arm))
    save_json(folder/'complete.json',dict(rows=rows,sha256=hashes,seconds=time.perf_counter()-tick))
    return rows


def fold(root,out,data,held,name,seed,device,epochs,quick=False):
    sources=[m for m in MATCHES if m!=held]
    train,cal,records=partitions(data,sources)
    folder=out/f'seed{seed}'/('leave_out_'+held)/name
    folder.mkdir(parents=True,exist_ok=True)
    tcn=get_model(folder/'model',name,seed,train,cal,records,sources,device,epochs)
    a,d=motion_pair(root,held); gates=get_gates(folder,tcn,a,d,cal,sources)
    rows=[]
    cases_spec=SCENARIOS[:1] if quick else SCENARIOS+[('slots',None,1000,False),('noise',None,1000,False)]
    for scenario,fraction,mask_seed,long in cases_spec:
        dest=folder/f'{scenario}_{mask_seed}'
        if (dest/'complete.json').exists():
            marker=read_json(dest/'complete.json')
            if any(sha256(dest/n)!=h for n,h in marker['sha256'].items()): raise ValueError('Changed prediction')
            rows.extend(marker['rows']); continue
        dest.mkdir(exist_ok=True)
        cases=[]
        for i,(original,y) in enumerate(data[held]):
            s=mask_seed+10*MATCHES.index(held)+i
            if scenario in ('slots','noise'):
                changed=perturb(original,'slots' if scenario=='slots' else 'noise_0.3m',s)[0]
            else: changed=original if fraction is None else mask_ball(original,fraction,s,long)[0]
            cases.append((changed,y))
        rows.extend(score_cases(tcn,a,d,gates,cases,data[held],dest,
            dict(match=held,recipe=name,train_seed=seed,scenario=scenario,seed=mask_seed)))
        print('EVALUATED',held,name,seed,scenario,mask_seed,flush=True)
    return rows


def decision(rows,seed):
    selected=[dict(r,candidate='tcn' if r['candidate']=='geometry__plain' else r['candidate'])
              for r in rows if r['train_seed']==seed and r['scenario'] not in ('slots','noise')]
    names=tuple(dict.fromkeys(r['candidate'] for r in selected))
    _,means,result=summarize(selected,names)
    # Do not carry legacy bookkeeping fields into the new public protocol.
    result.pop('organizer_labels_used',None)
    return result,means


def write_tables(out,rows):
    import csv
    save_json(out/'scores.json',rows)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'scores.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def run(args):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.device=='cuda' and not torch.cuda.is_available(): raise ValueError('Select a GPU runtime')
    manifest=verify(args.inputs)
    args.out.mkdir(parents=True,exist_ok=True)
    code=Path(__file__).resolve().parent
    protocol=dict(schema='final-round-v1',recipes=RECIPES,arms=ARMS,seeds=SEEDS,
        input_sha256=sha256(args.inputs/'manifest.json'),environment=environment(),
        code_sha256={p.name:sha256(p) for p in code.glob('*.py')},device=args.device,
        epochs=args.epochs,quick=args.quick,scorer_sha256=sha256(code.parent/'evaluation/possession_metrics.py'),
        full_matches=list(MATCHES),historical_opened_reserve=['J03WQQ','J03WR9'],
        stage='eight recipes five folds seed7, then two recipe families plus control seeds17/27',
        selection='baseline_suite guards; shortlist by seed7 stress, confirmation requires all three seeds',
        calibration='source tails shared by temperature and gate; whole evaluated match excluded',
        human_test_matches=['118577'])
    pp=args.out/'protocol_before_run.json'
    canonical=json.loads(json.dumps(protocol))
    if pp.exists() and read_json(pp)!=canonical: raise ValueError('Cannot resume with changed code/data/environment/options')
    save_json(pp,protocol)
    data=datasets(args.inputs);rows=[]
    names=list(RECIPES) if not args.quick else ['geometry']
    helds=list(MATCHES) if not args.quick else [MATCHES[0]]
    for held in helds:
        for name in names:
            rows.extend(fold(args.inputs,args.out,data,held,name,7,args.device,args.epochs,args.quick))
            write_tables(args.out,rows)
    if args.quick:
        save_json(args.out/'smoke_complete.json',dict(quality_claim=False,rows=len(rows)));return
    first,means=decision(rows,7);save_json(args.out/'screen_decision.json',first)
    eligible=sorted((c for c,v in first['eligibility'].items() if v['eligible']),
                    key=lambda c:first['eligibility'][c]['stress_spell'],reverse=True)
    shortlist=[]
    for candidate in eligible:
        name=candidate.split('__')[0]
        if name not in shortlist: shortlist.append(name)
        if len(shortlist)==2: break
    shortlist=list(dict.fromkeys(['geometry',*shortlist]))
    save_json(args.out/'shortlist_before_confirmation.json',dict(recipes=shortlist,eligible_arms=eligible))
    confirmations={7:first}
    for seed in SEEDS[1:]:
        for held in MATCHES:
            for name in shortlist:
                rows.extend(fold(args.inputs,args.out,data,held,name,seed,args.device,args.epochs))
                write_tables(args.out,rows)
        confirmations[seed],_=decision(rows,seed)
    common=[c for c in eligible if all(c in r['eligibility'] and r['eligibility'][c]['eligible']
                                      for r in confirmations.values())]
    selected=max(common,key=lambda c:np.mean([r['eligibility'][c]['stress_spell'] for r in confirmations.values()])) if common else 'tcn'
    save_json(args.out/'decision.json',dict(selected=selected,confirmations=confirmations,
        rule='Pass every original gate in each of three training seeds; highest mean stress score',
        independent_test=False,scope='development selection on repeated five-match OOF'))
    selected_recipe='geometry' if selected=='tcn' else selected.split('__')[0]
    selected_arm='plain' if selected=='tcn' else selected.split('__')[1]
    train,cal,records=partitions(data,list(MATCHES));a,d=motion_pair(args.inputs,None)
    # Export the control and the selected complete pipelines before any new test reference is read.
    for name in dict.fromkeys(['geometry',selected_recipe]):
        for seed in SEEDS:
            dest=args.out/'release'/name/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
            tcn=get_model(dest/'model',name,seed,train,cal,records,list(MATCHES),args.device,args.epochs)
            get_gates(dest,tcn,a,d,cal,list(MATCHES))
    for name in ('A','dead_specialist'):
        dst=args.out/'release'/name
        if not dst.exists(): shutil.copytree(args.inputs/'experts/final'/name,dst)
    save_json(args.out/'release/frozen.json',dict(recipe=selected_recipe,arm=selected_arm,seeds=SEEDS,
        selection_sha256=sha256(args.out/'decision.json'),training_matches=list(MATCHES),
        deployment_seed=7,seed_replicates='Robustness evidence; no unvalidated seed averaging',
        sha256={p.relative_to(args.out/'release').as_posix():sha256(p) for p in (args.out/'release').rglob('*') if p.is_file() and p.name!='frozen.json'}))
    save_json(args.out/'complete.json',dict(rows=len(rows),selected=selected,release_sha256=sha256(args.out/'release/frozen.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--quick',action='store_true',help='Full-match software smoke only, never model selection')
    run(p.parse_args())
