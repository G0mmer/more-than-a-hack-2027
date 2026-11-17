"""Five-match public DFL cross-validation with matched CRF calibration."""
from .external_paths import allowed_output
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np

from .data import sha256
from .evaluation import official
from .probabilistic import STRIDE, split_ranges, transition_parameters, fit_temperature, temperature
from .probabilistic_run import fusion
from .robust import VARIANTS, RobustCRF, Bundle, crop, mask_ball, sequence
from .robust_run import read_match, score
from .models import features, baseline, rules, RuleConfig
from .probabilistic import probability_metrics
from .run import save_json, save_table

MATCHES = ('J03WMX', 'J03WN1', 'J03WPY', 'J03WOH', 'J03WOY')
RESERVED = ('J03WQQ', 'J03WR9')
BASELINES = ('rules_V0', 'B50')
SEEDS = (1000, 1001, 1002)
SCENARIOS = [('clean', None, 0, False)] + [
    (name, fraction, seed, long)
    for name, fraction, long in (('missing20', .2, False), ('missing40', .4, False), ('long40', .4, True))
    for seed in SEEDS
]
METRICS = ('spell_f1', 'macro_f1', 'f1_D', 'takeover_25_f1', 'stoppage_25_f1', 'nll', 'ece', 'brier')


def protocol():
    return {
        'version': 'dfl-multimatch-v1', 'matches': list(MATCHES), 'reserved_unopened': list(RESERVED),
        'folds': [{'validation': m, 'training': [s for s in MATCHES if s != m]} for m in MATCHES],
        'variants': VARIANTS, 'baselines': BASELINES,
        'split': 'First 80% of each training half for fitting, final 20% for calibration; 125 frames removed on each side of split.',
        'training': 'Clean halves; augmented variants also receive a masked40 copy. Same match-specific masks across folds and variants.',
        'calibration': 'All variants, including control: identical clean + masked40 calibration halves. Fit emission temperature, then marginal temperature.',
        'parameters': {'c2': .1, 'context_samples': 5, 'chain_strength': .25, 'stride': STRIDE, 'iterations': 80},
        'scenarios': SCENARIOS,
        'mask_blocks': 'Training/calibration and missing20/40: nominal 0.52-2 s. Long40: nominal 2-8 s. Overlap can create longer contiguous gaps.',
        'seeds': 'Training: 7+10*match_index+half_index; calibration: 103+10*match_index+half_index; evaluation: scenario seed+10*match_index+half_index.',
        'selection': {
            'unit': 'Each validation match equally weighted; seeds averaged within match and scenario.',
            'eligible': 'Clean mean delta >= -0.01 and every clean match delta >= -0.03 vs control; mean stress delta > 0 and positive on >=4 of 5 matches.',
            'stress': 'Equal mean of missing40 and long40 spellF1.',
            'tie_break': 'Within 0.01 of best eligible stress score prefer fewer history features, then no augmentation. Control is always eligible.',
            'final': 'Fit all four variants on the five development matches with the same 80/20 procedure. Selected variant is frozen from cross-validation; final training metrics do not select it.'},
        'limitations': ['Five development matches; overlapping training folds are dependent.',
                       'Synthetic missingness does not reproduce every tracking failure.',
                       'DFL BallStatus/BallPossession define the reference.',
                       'Calibration tails share matches with fitting; whole validation match is excluded from both.',
                       'Final refits have no independent score in this experiment. CV scores estimate a variant, not the exact refitted weights.']}


def partitions(datasets, sources):
    """Only explicitly listed sources may contribute to fitting or calibration."""
    if not sources or len(set(sources)) != len(sources) or not set(sources) <= set(MATCHES):
        raise ValueError('Invalid training match list')
    train, cal, records = [], [], []
    for match in sources:
        for half_index, (d, y) in enumerate(datasets[match]):
            ranges = split_ranges(len(y))
            a, b = ranges['refit']
            c, e = ranges['calibrate']
            seed_offset = 10 * MATCHES.index(match) + half_index
            train.append((crop(d, a, b), y[a:b], 7 + seed_offset))
            cal.append((crop(d, c, e), y[c:e], 103 + seed_offset))
            records.append({'match': match, 'half': d.period, 'fit': [a,b], 'calibrate': [c,e],
                            'fit_frames': [int(d.frame_ids[a]), int(d.frame_ids[b-1])],
                            'calibration_frames': [int(d.frame_ids[c]), int(d.frame_ids[e-1])]})
    return train, cal, records


def fit_bundle(datasets, sources, name, folder, pm):
    history, augmented = VARIANTS[name]
    train, cal, records = partitions(datasets, sources)
    clean = [sequence(d, history, y) for d, y, seed in train]
    transition, prior = transition_parameters(clean)
    training = clean + ([sequence(mask_ball(d,.4,seed)[0],history,y) for d,y,seed in train] if augmented else [])
    print('FIT ' + folder.parent.name + '/' + name, flush=True)
    model = RobustCRF(history).fit(training, folder)
    cases = [(d,y) for d,y,seed in cal] + [(mask_ball(d,.4,seed)[0],y) for d,y,seed in cal]
    seqs = [sequence(d,history,y) for d,y in cases]
    raw = [model.predict(s)[0] for s in seqs]
    truth = np.concatenate([s.y for s in seqs])
    emission_temp = fit_temperature(np.concatenate(raw), truth)
    chained = [fusion([temperature(p,emission_temp)],[1.],transition,prior,.25)[0] for p in raw]
    chain_temp = fit_temperature(np.concatenate(chained),truth)
    config = {'history':history, 'augmentation':augmented, 'variant':name, 'stride':STRIDE,
              'emission_temperature':emission_temp, 'chain_temperature':chain_temp,
              'transition':transition.tolist(), 'prior':prior.tolist(), 'c2':.1,
              'context':5, 'chain_strength':.25, 'training_matches':list(sources),
              'calibration':'source tails: clean + missing40; same protocol for all variants',
              'partitions':records}
    save_json(folder/'bundle.json', config)
    bundle = Bundle(model,config)
    save_json(folder/'complete.json', {'artifact_sha256':{f:sha256(folder/f) for f in
              ('bundle.json','model.crfsuite','scaler.npz')}})
    print('CALIBRATED ' + folder.parent.name + '/' + name, flush=True)
    return bundle


def load_completed(folder):
    marker = folder/'complete.json'
    if not marker.is_file():
        raise ValueError('Incomplete model folder: '+str(folder)+'. Use a new output directory.')
    state = json.loads(marker.read_text(encoding='utf-8'))
    for file, checksum in state['artifact_sha256'].items():
        if sha256(folder/file) != checksum:
            raise ValueError('Model artifact changed: '+str(folder/file))
    return Bundle.load(folder)


def evaluate_fold(match, halves, bundles, dest, pm):
    rows = []
    for scenario, fraction, seed, long in SCENARIOS:
        path = dest/(scenario+'_'+str(seed)+'.json')
        if path.exists():
            rows.extend(json.loads(path.read_text(encoding='utf-8')))
            continue
        cases, rates = [], []
        for i, (d,y) in enumerate(halves):
            altered, rate = (d,0.) if fraction is None else mask_ball(d,fraction,seed+10*MATCHES.index(match)+i,long)
            cases.append((altered,y)); rates.append(rate)
        batch = []
        for name in (*VARIANTS,*BASELINES):
            if name in bundles:
                pred = [bundles[name].predict(d) for d,y in cases]
                ys = [y for p,y in pred]
                metrics = probability_metrics(np.concatenate([p for p,y in pred]),np.concatenate([y for d,y in cases]),np.concatenate(ys))
            else:
                ys = [baseline(features(d),50) if name=='B50' else rules(features(d),RuleConfig(.2,.08,.4,True)) for d,y in cases]
                metrics = {}
            row = score(pm,cases,ys)
            row.update(match=match,variant=name,scenario=scenario,seed=seed,
                       masked_fraction=float(np.mean(rates)),**{k:metrics[k] for k in ('nll','ece','brier') if k in metrics})
            batch.append(row)
            print(f"EVAL {match} {scenario}/{seed} {name}: {row['spell_f1']:.4f}",flush=True)
        save_json(path,batch)
        rows.extend(batch)
    return rows


def summarize(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault((r['match'],r['variant'],r['scenario']),[]).append(r)
    expected = {(m,v,s) for m in MATCHES for v in (*VARIANTS,*BASELINES) for s in ('clean','missing20','missing40','long40')}
    if set(grouped) != expected:
        raise ValueError('Incomplete CV result grid')
    per_match = []
    for (match,name,scenario), values in grouped.items():
        seeds = [r['seed'] for r in values]
        if sorted(seeds) != ([0] if scenario=='clean' else sorted(SEEDS)):
            raise ValueError('Missing or duplicate evaluation seed')
        r = {'match':match,'variant':name,'scenario':scenario,'seeds':len(values)}
        r.update({metric:float(np.mean([v[metric] for v in values])) for metric in METRICS if metric in values[0]})
        per_match.append(r)
    means = []
    for name in (*VARIANTS,*BASELINES):
        for scenario in ('clean','missing20','missing40','long40'):
            values = [r for r in per_match if r['variant']==name and r['scenario']==scenario]
            r = {'variant':name,'scenario':scenario,'matches':len(values)}
            for metric in METRICS:
                if metric in values[0]: r[metric]=float(np.mean([v[metric] for v in values]))
            means.append(r)
    return per_match,means


def select_variant(per_match):
    values = {(r['match'],r['variant'],r['scenario']):r['spell_f1'] for r in per_match}
    def scores(name,scenario):
        return np.array([values[m,name,scenario] for m in MATCHES])
    def stress(name):
        return (scores(name,'missing40')+scores(name,'long40'))/2
    decisions = []
    for name in VARIANTS:
        clean_delta = scores(name,'clean')-scores('control','clean')
        stress_delta = stress(name)-stress('control')
        eligible = name=='control' or (clean_delta.mean()>=-.01 and clean_delta.min()>=-.03
                    and stress_delta.mean()>0 and (stress_delta>0).sum()>=4)
        decisions.append({'variant':name,'eligible':bool(eligible),'clean_delta_mean':float(clean_delta.mean()),
                          'clean_delta_worst':float(clean_delta.min()),'stress_delta_mean':float(stress_delta.mean()),
                          'stress_wins':int((stress_delta>0).sum()),'stress_spell_f1':float(stress(name).mean())})
    eligible = [r for r in decisions if r['eligible']]
    best = max(r['stress_spell_f1'] for r in eligible)
    tied = [r for r in eligible if r['stress_spell_f1']>=best-.01]
    winner = min(tied,key=lambda r:(VARIANTS[r['variant']][0],VARIANTS[r['variant']][1],-r['stress_spell_f1']))
    return {'selected':winner['variant'],'decisions':decisions,'rule':protocol()['selection']}


def write_report(out, per_match, means, selection):
    lines = ['# Five-match DFL CRF experiment','',
             'Four CRF variants use identical source-only calibration. Whole validation matches are excluded from fitting and calibration.',
             'Means weight matches equally; perturbation seeds are averaged within each match. Folds are dependent.',
             'Control is retrained here, so it differs from the previous frozen single-match control.','',
             'Selected from the declared rule: **'+selection['selected']+'**.','',
             '| Variant | Clean | Missing20 | Missing40 | Long40 |','|---|---:|---:|---:|---:|']
    lookup = {(r['variant'],r['scenario']):r['spell_f1'] for r in means}
    for name in (*VARIANTS,*BASELINES):
        lines.append('| '+name+' | '+' | '.join(f'{lookup[name,s]:.4f}' for s in ('clean','missing20','missing40','long40'))+' |')
    lines += ['','## Per-match spellF1','','| Match | Variant | Clean | Missing40 | Long40 |','|---|---|---:|---:|---:|']
    lookup = {(r['match'],r['variant'],r['scenario']):r['spell_f1'] for r in per_match}
    for match in MATCHES:
        for name in (*VARIANTS,*BASELINES):
            lines.append('| '+match+' | '+name+' | '+' | '.join(f'{lookup[match,name,s]:.4f}' for s in ('clean','missing40','long40'))+' |')
    lines += ['','## Limitations','']+['- '+s for s in protocol()['limitations']]
    lines += ['','Reserved J03WQQ/J03WR9 were not opened. final/ contains refits on all five development matches; their CV references are estimates for the variants, not independent scores for those exact refits.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dfl-root',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--resume',action='store_true',help='Resume only the same protocol, inputs and implementation')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    out=args.out.resolve()
    if (not allowed_output(out, repo)) or (out.exists() and not args.resume) or (args.resume and not out.is_dir()):
        parser.error('Use a NEW output folder under data/external/runs or outside repo, or --resume for an existing run')
    out.mkdir(parents=True,exist_ok=True)
    code={str(p.relative_to(repo)).replace('\\','/'):sha256(p) for p in (repo/'possession_lab').glob('*.py')}
    current={'protocol':protocol(),'code_sha256':code,'scorer_sha256':sha256(repo/'evaluation/possession_metrics.py'),
             'dfl_root':str(args.dfl_root.resolve())}
    # JSON normalization converts tuples before comparison on resume.
    current=json.loads(json.dumps(current))
    lock=out/'protocol_before_run.json'
    if args.resume:
        if json.loads(lock.read_text(encoding='utf-8'))!=current:
            raise ValueError('Protocol/code/root changed; use a new output folder')
    else:
        save_json(lock,current)
    datasets,audits,files={},{},[]
    for match in MATCHES:
        print('READ PUBLIC DFL '+match,flush=True)
        datasets[match],audits[match],paths=read_match(args.dfl_root,match)
        files.extend(paths)
    inputs=out/'inputs.json'
    if args.resume and inputs.exists() and json.loads(inputs.read_text(encoding='utf-8'))!=files:
        raise ValueError('Input files changed')
    save_json(inputs,files); save_json(out/'audits.json',audits)
    pm=official(repo)
    rows=[]
    for target in MATCHES:
        sources=[s for s in MATCHES if s!=target]
        fold=out/('leave_out_'+target)
        fold.mkdir(exist_ok=True)
        bundles={}
        for name in VARIANTS:
            folder=fold/name
            bundles[name]=load_completed(folder) if folder.exists() else fit_bundle(datasets,sources,name,folder,pm)
        rows.extend(evaluate_fold(target,datasets[target],bundles,fold,pm))
        save_json(out/'partial_scores.json',rows)
    per_match,means=summarize(rows)
    selection=select_variant(per_match)
    save_json(out/'selection_before_refit.json',selection)
    save_json(out/'scores.json',rows); save_json(out/'per_match.json',per_match); save_json(out/'means.json',means)
    save_table(out/'per_match.csv',per_match,['match','variant','scenario','seeds',*METRICS])
    save_table(out/'means.csv',means,['variant','scenario','matches',*METRICS])
    for name in VARIANTS:
        folder=out/'final'/name
        if folder.exists(): load_completed(folder)
        else: fit_bundle(datasets,list(MATCHES),name,folder,pm)
    save_json(out/'manifest.json',{'version':'dfl-multimatch-v1','sources':files,'selected':selection['selected'],
        'code_sha256':code,'scorer_sha256':current['scorer_sha256'],
        'model_sha256':{str(p.relative_to(out)).replace('\\','/'):sha256(p) for p in out.glob('final/*/*') if p.is_file()},
        'utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,
        'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas','python-crfsuite')},
        'reserved_unopened':list(RESERVED)})
    write_report(out,per_match,means,selection)
    print('COMPLETE '+str(out),flush=True)


if __name__=='__main__':
    main()
