"""Fixed baseline training and external comparison.

Run --help for the independent commands. Competition labels never enter
training or the external decision. Existing frozen experiment files are read-only.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from .data import Tracking, sha256
from .ensemble import CANDIDATES, MotionBundle, predict_candidates
from .evaluation import official, evaluate_period, restart_counts
from .external_paths import allowed_output
from .multimatch_run import MATCHES, RESERVED, SCENARIOS, partitions
from .probabilistic import probability_metrics
from .robust import mask_ball
from .robust_run import read_match, score
from .run import save_json, save_table
from .team_temporal import TeamConfig, TeamTemporalBundle, fit
from .team_temporal_run import prepared_views
from .transformer_temporal import C3Bundle

REPO = Path(__file__).resolve().parents[1]
METRICS = ('spell_f1', 'macro_f1', 'f1_D', 'restart_recall', 'nll', 'ece', 'brier')


def environment():
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages={p: importlib.metadata.version(p) for p in
                          ('numpy', 'pandas', 'scipy', 'torch', 'python-crfsuite')})


def initialize(out, command, metadata):
    out = out.resolve()
    if out.exists() or not allowed_output(out, REPO):
        raise ValueError('Use a NEW output outside the repository or under data/external/runs')
    out.mkdir(parents=True)
    save_json(out/'protocol_before_run.json', dict(schema='baseline-suite-v1', command=command,
              utc=datetime.now(timezone.utc).isoformat(), environment=environment(),
              code_sha256={str(p.relative_to(REPO)): sha256(p) for p in
                           sorted((REPO/'possession_lab').glob('*.py'))},
              scorer_sha256=sha256(REPO/'evaluation/possession_metrics.py'), **metadata))
    for directory in ('possession_lab', 'evaluation'):
        for source in (REPO/directory).glob('*.py'):
            target=out/'source_snapshot'/directory/source.name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)


def restart_recall(pm, cases, predictions):
    tp = fn = 0
    for (_, truth), pred in zip(cases, predictions):
        matched, missed = restart_counts(pm, truth, pred)
        tp += matched
        fn += missed
    return tp/(tp+fn) if tp+fn else 1.


def summarize(rows, candidates=CANDIDATES):
    expected = {(m, c, s, seed) for m in MATCHES for c in candidates
                for s, _, seed, _ in SCENARIOS}
    keys = [(r['match'], r['candidate'], r['scenario'], r['seed']) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Incomplete or duplicate external evaluation grid')
    per = []
    scenarios = ('clean', 'missing20', 'missing40', 'long40')
    for match in MATCHES:
        for candidate in candidates:
            for scenario in scenarios:
                group = [r for r in rows if (r['match'], r['candidate'], r['scenario']) ==
                         (match, candidate, scenario)]
                per.append(dict(match=match, candidate=candidate, scenario=scenario,
                                **{k: float(np.mean([r[k] for r in group])) for k in METRICS}))
    means = [dict(candidate=c, scenario=s, **{k: float(np.mean([r[k] for r in per
              if r['candidate'] == c and r['scenario'] == s])) for k in METRICS})
             for c in candidates for s in scenarios]
    def metric(candidate, scenario, key):
        return next(r[key] for r in means if (r['candidate'], r['scenario']) == (candidate, scenario))
    eligibility = {}
    for c in candidates:
        deltas = {(r['match'], r['scenario']): {k: r[k]-next(b[k] for b in per
                  if b['match'] == r['match'] and b['scenario'] == r['scenario'] and b['candidate'] == 'tcn')
                  for k in METRICS} for r in per if r['candidate'] == c}
        stress = np.mean([metric(c, s, 'spell_f1') for s in ('missing40', 'long40')])
        baseline = np.mean([metric('tcn', s, 'spell_f1') for s in ('missing40', 'long40')])
        checks = dict(clean_mean=metric(c, 'clean', 'spell_f1') >= metric('tcn', 'clean', 'spell_f1')-.005,
                      clean_each=all(deltas[(m, 'clean')]['spell_f1'] >= -.03 for m in MATCHES),
                      stress_mean=stress > baseline,
                      stress_four_matches=sum(np.mean([deltas[(m,s)]['spell_f1'] for s in ('missing40','long40')]) > 0
                                              for m in MATCHES) >= 4,
                      dead_stress=all(metric(c,s,'f1_D') >= metric('tcn',s,'f1_D')-.005
                                      for s in ('missing40','long40')),
                      restart_mean=all(metric(c,s,'restart_recall') >= metric('tcn',s,'restart_recall')-.02
                                       for s in scenarios),
                      restart_each=all(d['restart_recall'] >= -.05 for d in deltas.values()))
        eligibility[c] = dict(eligible=bool(all(checks.values())), checks={k:bool(v) for k,v in checks.items()},
                              stress_spell=float(stress))
    eligible = [c for c in candidates if c != 'tcn' and eligibility[c]['eligible']]
    selected = max(eligible, key=lambda c: eligibility[c]['stress_spell']) if eligible else 'tcn'
    return per, means, dict(schema='baseline-ensemble-decision-v1', selected=selected,
                           eligibility=eligibility, scope='development OOF selection; not an independent final test',
                           organizer_labels_used=False)


def compare(args):
    initialize(args.out, 'compare', dict(candidates=list(CANDIDATES), scenarios=SCENARIOS,
               held_out_matches=list(MATCHES), reserved_unopened=list(RESERVED),
               selection_rule='clean mean >=TCN-.005, each >=TCN-.03; stress gain >0 and 4/5 wins; '
               'D stress mean >=TCN-.005; restart recall mean >=TCN-.02 and each >=TCN-.05; '
               'maximize equal mean missing40/long40 among eligible, otherwise keep TCN',
               combination='calibrated native emissions at 12.5Hz; one TCN decoder for mixtures; '
               'native decoder for A controls; .5/.5 mean; gate uses TCN when ball missing; '
               'D replaces only D mass and preserves conditional H/A',
               temporal_release=str(args.temporal_release.resolve()), tracking_run=str(args.tracking_run.resolve())))
    pm, rows = official(), []
    for match in MATCHES:
        print('READ EXTERNAL '+match, flush=True)
        halves, audit, files = read_match(args.dfl_root, match)
        save_json(args.out/(match+'_input_audit.json'), dict(audit=audit, sources=files))
        folder = args.tracking_run/'folds'/('leave_out_'+match)
        a, d = MotionBundle(folder/'A'), MotionBundle(folder/'dead_specialist', True)
        tcn = C3Bundle.load(args.temporal_release/'oof_models'/('leave_out_'+match)/'tcn')
        sources = [m for m in MATCHES if m != match]
        if tcn.config['training_matches'] != sources or a.config['fold']['sources'] != sources or d.config['fold']['sources'] != sources:
            raise ValueError('Held-out match leaked into fitting/calibration')
        fold = args.out/('leave_out_'+match)
        fold.mkdir()
        for scenario, fraction, seed, long in SCENARIOS:
            cases = [(data if fraction is None else mask_ball(data, fraction,
                      seed+10*MATCHES.index(match)+i, long)[0], y) for i,(data,y) in enumerate(halves)]
            predictions = [predict_candidates(tcn, a, d, data) for data,_ in cases]
            for candidate in CANDIDATES:
                ps, ys = zip(*(p[candidate] for p in predictions))
                row = score(pm, cases, ys)
                row.update(probability_metrics(np.concatenate(ps), np.concatenate([y for _,y in cases]), np.concatenate(ys)))
                row.update(match=match, candidate=candidate, scenario=scenario, seed=seed,
                           restart_recall=restart_recall(pm,cases,ys))
                rows.append(row)
                for (data,_), y in zip(cases,ys):
                    # Labels permit rescoring; no raw external tracking is copied.
                    np.savez_compressed(fold/f'{candidate}_{scenario}_{seed}_{data.period}.npz',
                                        frame=data.frame_ids, label=y, probability_stage='decoded')
            save_json(args.out/'scores_partial.json', rows)
            print(match, scenario, seed, {r['candidate']:round(r['spell_f1'],4) for r in rows[-len(CANDIDATES):]}, flush=True)
    per, means, decision = summarize(rows)
    save_json(args.out/'scores.json', rows)
    save_json(args.out/'means.json', means)
    save_table(args.out/'per_match.csv', per, ['match','candidate','scenario',*METRICS])
    save_json(args.out/'decision.json', decision)
    print(json.dumps(decision, indent=2), flush=True)


def train(args):
    initialize(args.out, 'train', dict(recipe=asdict(TeamConfig()), matches=list(MATCHES),
               reserved_unopened=list(RESERVED), split='source80/calibration20; 125-frame purge each side',
               training='clean+missing40; fixed 8 epochs; final epoch; seed7; CPU threads2',
               synthetic=args.synthetic))
    from .temporal_run import synthetic_datasets
    datasets = synthetic_datasets() if args.synthetic else {}
    sources = []
    if not args.synthetic:
        for match in MATCHES:
            print('READ EXTERNAL '+match, flush=True)
            datasets[match], _, files = read_match(args.dfl_root, match)
            sources.extend(files)
    save_json(args.out/'inputs.json', sources)
    save_json(args.out/'data_kind.json', {'synthetic':args.synthetic, 'football_quality_claim':False})
    runs = [([m for m in MATCHES if m != held], args.out/('leave_out_'+held)) for held in MATCHES]
    runs.append((list(MATCHES), args.out/'final_refit'))
    for matches, folder in runs:
        training, calibration, records = partitions(datasets,matches)
        prepared, calibrated, clean = prepared_views(training,calibration,False)
        fit(prepared,calibrated,clean,matches,records,folder,TeamConfig())
    save_json(args.out/'complete.json', dict(models=6, synthetic=args.synthetic,
              model_files={str(p.relative_to(args.out)):sha256(p) for p in sorted(args.out.rglob('model.pt'))}))


def evaluate(args):
    marker=json.loads((args.trained_run/'complete.json').read_text())
    initialize(args.out,'evaluate',dict(trained_run=str(args.trained_run.resolve()),
               train_manifest_sha256=sha256(args.trained_run/'protocol_before_run.json'),
               synthetic=marker['synthetic'],scenarios=SCENARIOS,reserved_unopened=list(RESERVED)))
    for name,digest in marker['model_files'].items():
        source=args.trained_run/name
        if not source.resolve().is_relative_to(args.trained_run.resolve()) or sha256(source)!=digest:
            raise ValueError('Changed training weights')
    from .temporal_run import synthetic_datasets
    datasets=synthetic_datasets() if marker['synthetic'] else None
    pm,rows=official(),[]
    for match in MATCHES:
        if datasets is not None:
            halves=datasets[match]
        else:
            halves,_,files=read_match(args.dfl_root,match)
            save_json(args.out/(match+'_inputs.json'),files)
        model=TeamTemporalBundle.load(args.trained_run/('leave_out_'+match))
        if model.config['training_matches']!=[m for m in MATCHES if m!=match]:
            raise ValueError('Held-out match in training sources')
        for scenario,fraction,seed,long in SCENARIOS:
            cases=[(data if fraction is None else mask_ball(data,fraction,
                    seed+10*MATCHES.index(match)+i,long)[0],y) for i,(data,y) in enumerate(halves)]
            predictions=[model.predict(data) for data,_ in cases]
            ps,ys=zip(*predictions)
            row=score(pm,cases,ys)
            row.update(probability_metrics(np.concatenate(ps),np.concatenate([y for _,y in cases]),np.concatenate(ys)))
            row.update(match=match,scenario=scenario,seed=seed,restart_recall=restart_recall(pm,cases,ys))
            rows.append(row)
        print('EVALUATED '+match,flush=True)
        save_json(args.out/'scores_partial.json',rows)
    if len(rows)!=50:raise ValueError('Incomplete evaluation')
    per=[]
    for match in MATCHES:
        for scenario in ('clean','missing20','missing40','long40'):
            group=[r for r in rows if r['match']==match and r['scenario']==scenario]
            per.append(dict(match=match,scenario=scenario,**{k:float(np.mean([r[k] for r in group])) for k in METRICS}))
    means=[dict(scenario=s,**{k:float(np.mean([r[k] for r in per if r['scenario']==s])) for k in METRICS})
           for s in ('clean','missing20','missing40','long40')]
    save_json(args.out/'scores.json',rows)
    save_json(args.out/'means.json',means)
    save_table(args.out/'per_match.csv',per,['match','scenario',*METRICS])
    print(json.dumps(means,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('compare',help='Seven fixed candidates; five external held-out matches; no refitting')
    p.add_argument('--temporal-release',type=Path,required=True)
    p.add_argument('--tracking-run',type=Path,required=True)
    p.add_argument('--dfl-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p=commands.add_parser('evaluate',help='Score all five held-out TCN models from a fixed training run')
    p.add_argument('--trained-run',type=Path,required=True)
    p.add_argument('--dfl-root',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p=commands.add_parser('train',help='One fixed TCN recipe, five folds and final refit')
    p.add_argument('--dfl-root',type=Path)
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='train' and not args.synthetic and args.dfl_root is None:
        parser.error('--dfl-root is required for real-data training')
    if args.command=='evaluate' and args.dfl_root is None:
        marker=json.loads((args.trained_run/'complete.json').read_text())
        if not marker['synthetic']:parser.error('--dfl-root is required for real-data evaluation')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    {'compare':compare,'train':train,'evaluate':evaluate}[args.command](args)


if __name__=='__main__':
    main()
