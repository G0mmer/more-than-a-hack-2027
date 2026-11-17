"""C2 external DFL ablations: context first, then fixed-context player pooling."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .data import sha256
from .evaluation import official
from .external_paths import allowed_output
from .multimatch_run import MATCHES, RESERVED, SCENARIOS, partitions, load_completed
from .probabilistic import probability_metrics
from .robust import mask_ball
from .robust_run import read_match, score
from .run import save_json, save_table
from .temporal_run import METRICS, completed_case, save_prediction, synthetic_datasets
from .team_temporal import DILATIONS, TeamConfig, TeamTemporalBundle, prepare, fit


VARIANTS = {'context_2s': TeamConfig(context_seconds=2),
            'context_8s': TeamConfig(context_seconds=8),
            'context_32s': TeamConfig(context_seconds=32),
            'set_none_8s': TeamConfig(representation='none'),
            'set_mean_8s': TeamConfig(representation='mean'),
            'set_deepsets_8s': TeamConfig(representation='deepsets'),
            'set_graph_8s': TeamConfig(representation='graph')}
COMPARATORS = {'context_8s': 'context_2s', 'context_32s': 'context_2s',
               'set_none_8s': 'context_8s', 'set_mean_8s': 'set_none_8s',
               'set_deepsets_8s': 'set_mean_8s', 'set_graph_8s': 'set_deepsets_8s'}
SHARED_CODE = ('data.py', 'probabilistic.py', 'probabilistic_run.py', 'robust.py',
               'robust_run.py', 'multimatch_run.py', 'evaluation.py', 'models.py')


def row_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_case(path, models, match, scenario, seed, exports):
    record = json.loads(path.read_text())
    if row_digest(record['rows']) != record['rows_sha256']:
        raise ValueError('C2 score record changed')
    keys = [(r['variant'], r['match'], r['scenario'], r['seed']) for r in record['rows']]
    if set(keys) != {(v, match, scenario, seed) for v in models} or len(keys) != len(models):
        raise ValueError('C2 case identity mismatch')
    if len(record['prediction_sha256']) != exports:
        raise ValueError('Missing C2 prediction exports')
    for name, digest in record['prediction_sha256'].items():
        if Path(name).name != name or sha256(path.parent/name) != digest:
            raise ValueError('C2 prediction export changed')
    return record['rows']


def summarize(rows, held, models):
    keys = [(r['match'], r['variant'], r['scenario'], r['seed']) for r in rows]
    expected = {(m, v, s, seed) for m in held for v in models for s, _, seed, _ in SCENARIOS}
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError('Incomplete or duplicate C2 evaluation grid')
    per = []
    for match in held:
        for name in models:
            for scenario in dict.fromkeys(s[0] for s in SCENARIOS):
                group = [r for r in rows if (r['match'], r['variant'], r['scenario']) == (match, name, scenario)]
                per.append(dict(match=match, variant=name, scenario=scenario, seeds=len(group),
                                **{k: float(np.mean([r[k] for r in group])) for k in METRICS}))
    paired = []
    for r in per:
        comparisons = ([COMPARATORS[r['variant']]] if COMPARATORS.get(r['variant']) in models else [])
        if r['variant'] in VARIANTS and 'crf_c1' in models:
            comparisons.append('crf_c1')
        for control in comparisons:
            base = next(b for b in per if b['match'] == r['match'] and b['scenario'] == r['scenario'] and b['variant'] == control)
            paired.append(dict(match=r['match'], variant=r['variant'], comparator=control, scenario=r['scenario'],
                               **{k+'_delta': r[k]-base[k] for k in METRICS}))
    return per, paired


def report(out, rows, held, models, synthetic, costs):
    per, paired = summarize(rows, held, models)
    save_table(out/'per_match.csv', per, ['match', 'variant', 'scenario', 'seeds', *METRICS])
    save_table(out/'paired_deltas.csv', paired, ['match', 'variant', 'comparator', 'scenario', *[k+'_delta' for k in METRICS]])
    save_json(out/'costs.json', costs)
    means = [dict(variant=v, scenario=s, matches=len(held),
                  **{k: float(np.mean([r[k] for r in per if r['variant'] == v and r['scenario'] == s])) for k in METRICS})
             for v in models for s in dict.fromkeys(x[0] for x in SCENARIOS)]
    save_json(out/'means.json', means)
    lines = ['# C2: temporal context and player sets', '',
             'SYNTHETIC SMOKE TEST. No football accuracy claim.' if synthetic else f'{len(held)} of five complete development folds.',
             'Context variants have identical capacity and initialization. Pooling variants have identical capacity and fixed nominal 8s context.',
             'Player-set variants omit slot-based nearby_speed in the shared geometry. set_none_8s controls that omission.',
             'Means weight matches equally after averaging the three masking seeds. No held-out model/epoch selection.', '',
             '| Model | Clean | Missing20 | Missing40 | Long40 |', '|---|---:|---:|---:|---:|']
    for name in models:
        values = [next(r['spell_f1'] for r in means if r['variant'] == name and r['scenario'] == s)
                  for s in ('clean', 'missing20', 'missing40', 'long40')]
        lines.append('| '+name+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines += ['', 'Detailed paired contrasts: paired_deltas.csv. Individual matches and probability/boundary metrics: per_match.csv.',
              'C1 controls are reused only after input, shared-code, checkpoint, partition and prediction-hash checks.',
              'Their timings come from the earlier run; hardware/load timing differences are not controlled.',
              'New inference timings include preparation, emissions and decoding, excluding scoring/export/XML.',
              'CPU/GPU device, fitting time and parameter count are in costs.json; unmeasured devices are not estimated.',
              'These are dependent development folds and synthetic gaps. Reserved matches and competition data remain unopened.',
              'No C4 final model is selected or refitted by this experiment.', '']
    (out/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def reference_manifest(folder, repo):
    manifest = json.loads((folder/'manifest.json').read_text())
    protocol = manifest['protocol']
    if protocol['synthetic'] or protocol['held'] != list(MATCHES) or protocol['crf_variant'] != 'augmentation':
        raise ValueError('Expected completed five-fold external C1 comparison')
    for name in SHARED_CODE:
        relative = 'possession_lab/'+name
        if sha256(repo/relative) != protocol['code_sha256'][relative]:
            raise ValueError('Shared C1 feature/split/metric code changed: '+name)
    if sha256(repo/'evaluation/possession_metrics.py') != protocol['scorer_sha256']:
        raise ValueError('Official scorer changed')
    for name, digest in manifest['artifact_sha256'].items():
        if not (folder/name).resolve().is_relative_to(folder.resolve()) or sha256(folder/name) != digest:
            raise ValueError('C1 reference artifact changed')
    return manifest


def prepared_views(train, cal, player_sets):
    clean = [prepare(d, player_sets, y) for d, y, _ in train]
    training = clean+[prepare(mask_ball(d, .4, seed)[0], player_sets, y) for d, y, seed in train]
    calibration = ([prepare(d, player_sets, y) for d, y, _ in cal] +
                   [prepare(mask_ball(d, .4, seed)[0], player_sets, y) for d, y, seed in cal])
    return training, calibration, clean


def run(args):
    repo = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    if (not allowed_output(out, repo) or (out.exists() and not args.resume) or
            (args.resume and not out.is_dir())):
        raise ValueError('Use a new output under data/external/runs or outside repo, or resume an existing run')
    if len(set(args.held)) != len(args.held) or len(set(args.variants)) != len(args.variants):
        raise ValueError('Duplicate match or variant')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    recipes = {name: replace(VARIANTS[name], epochs=args.epochs, hidden=args.hidden,
                            chunk_samples=args.chunk_samples, seed=args.seed) for name in args.variants}
    reference = reference_manifest(args.c1_run, repo) if args.c1_run else None
    models = list(recipes)+(['crf_c1', 'tcn_c1'] if reference else [])
    protocol = {'version': 'team-temporal-c2-v1', 'synthetic': args.synthetic, 'held': args.held,
                'recipes': {k: asdict(v) for k, v in recipes.items()}, 'dilations': DILATIONS,
                'hypotheses': ['Longer fixed-capacity context improves long-gap Spell F1.',
                               'Nonlinear set pooling or local relations improve over pooled means at fixed context.'],
                'comparators': COMPARATORS, 'scenarios': SCENARIOS,
                'split': 'Original source80/calibration20, 125-frame purge on both sides, held-out match excluded',
                'training': 'Clean + source-seeded missing40. Fixed epoch count from recipes. No held-out selection.',
                'architecture': 'Eight residual Conv1d/ReLU blocks, width22 by default; identical parameter budget across contexts',
                'calibration': 'Same clean/missing40 source tails, emission temperature and chain(.25) temperature',
                'player_set_features': 'Slot-free positions/ball/team offsets and nearest opponent; geometry column9 zero in all set variants',
                'graph': 'Separate teammate/opponent messages; exp(-distance/10m), cutoff20m, own self-edge',
                'reserved_unopened': list(RESERVED), 'dfl_root': None if args.synthetic else str(args.dfl_root.resolve()),
                'c1_run': None if not reference else str(args.c1_run.resolve()),
                'c1_manifest_sha256': None if not reference else sha256(args.c1_run/'manifest.json'),
                'device': args.device, 'threads': args.threads, 'platform': platform.platform(),
                'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas', 'torch', 'python-crfsuite')},
                'code_sha256': {str(p.relative_to(repo)): sha256(p) for p in (repo/'possession_lab').glob('*.py')},
                'scorer_sha256': sha256(repo/'evaluation/possession_metrics.py')}
    protocol = json.loads(json.dumps(protocol))
    if args.resume:
        if json.loads((out/'protocol_before_run.json').read_text()) != protocol:
            raise ValueError('C2 protocol, code, dependencies or reference changed')
    else:
        out.mkdir(parents=True)
        save_json(out/'protocol_before_run.json', protocol)
    datasets, audits, files = {}, {}, []
    if args.synthetic:
        datasets = synthetic_datasets()
    else:
        for match in MATCHES:
            print('READ EXTERNAL DFL '+match, flush=True)
            datasets[match], audits[match], sources = read_match(args.dfl_root, match)
            files.extend(sources)
    if reference:
        actual = {Path(p['path']).name: p['sha256'] for p in files}
        expected = {Path(p['path']).name: p['sha256'] for p in reference['sources']}
        if actual != expected:
            raise ValueError('C1/C2 data hashes differ')
    if args.resume and json.loads((out/'inputs.json').read_text()) != files:
        raise ValueError('C2 inputs changed')
    save_json(out/'inputs.json', files); save_json(out/'audits.json', audits)
    pm, rows, costs = official(repo), [], []
    for match in args.held:
        sources = [m for m in MATCHES if m != match]
        train, cal, records = partitions(datasets, sources)
        fold = out/('leave_out_'+match); fold.mkdir(exist_ok=True)
        if reference:
            control = load_completed(args.c1_run/fold.name/'crf')
            if control.config['partitions'] != records or control.config['training_matches'] != sources:
                raise ValueError('C1/C2 partition mismatch')
        packs, bundles = {}, {}
        for name, config in recipes.items():
            folder = fold/name
            if folder.exists():
                bundle = TeamTemporalBundle.load(folder, args.device)
            else:
                player_sets = config.representation != 'geometry'
                if player_sets not in packs:
                    packs[player_sets] = prepared_views(train, cal, player_sets)
                training, calibration, clean = packs[player_sets]
                bundle = fit(training, calibration, clean, sources, records, folder, config, args.device)
            if (bundle.config['partitions'] != records or bundle.config['training_matches'] != sources or
                    bundle.config['model'] != asdict(config)):
                raise ValueError('C2 checkpoint metadata mismatch')
            bundles[name] = bundle
            costs.append({k: bundle.config[k] for k in ('parameters', 'fit_seconds', 'calibration_seconds', 'device')} |
                         {'match': match, 'variant': name})
        del packs
        for scenario, fraction, seed, long in SCENARIOS:
            result = fold/f'{scenario}_{seed}.json'
            if result.exists():
                rows.extend(load_case(result, models, match, scenario, seed, 2*len(recipes)))
                continue
            cases = [(d if fraction is None else mask_ball(d, fraction, seed+10*MATCHES.index(match)+i, long)[0], y)
                     for i, (d, y) in enumerate(datasets[match])]
            cached, batch, hashes = {}, [], {}
            for name, bundle in bundles.items():
                player_sets = bundle.config['model']['representation'] != 'geometry'
                if player_sets not in cached:
                    started = time.perf_counter()
                    encoded = [prepare(d, player_sets) for d, _ in cases]
                    cached[player_sets] = (encoded, time.perf_counter()-started)
                encoded, feature_seconds = cached[player_sets]
                started = time.perf_counter()
                predictions = [bundle.predict_prepared(s) for s in encoded]
                elapsed = time.perf_counter()-started
                row = score(pm, cases, [y for _, y in predictions])
                prob = probability_metrics(np.concatenate([p for p, _ in predictions]), np.concatenate([y for _, y in cases]),
                                           np.concatenate([y for _, y in predictions]))
                row.update(match=match, variant=name, scenario=scenario, seed=seed,
                           feature_seconds=feature_seconds, inference_seconds=elapsed+feature_seconds,
                           **{k: prob[k] for k in ('nll', 'ece', 'brier')})
                for (d, _), (p, y) in zip(cases, predictions):
                    path = fold/f'{name}_{scenario}_{seed}_{d.period}.npz'
                    save_prediction(path, match, d, p, y); hashes[path.name] = sha256(path)
                batch.append(row)
                print(f"EVAL {match} {scenario}/{seed} {name}: {row['spell_f1']:.4f}", flush=True)
            if reference:
                for row in completed_case(args.c1_run/fold.name/result.name):
                    batch.append(row | {'variant': row['variant']+'_c1'})
            save_json(result, {'rows': batch, 'rows_sha256': row_digest(batch), 'prediction_sha256': hashes})
            rows.extend(batch)
        report(out, rows, args.held[:args.held.index(match)+1], models, args.synthetic, costs)
        save_json(out/'partial_scores.json', rows)
    save_json(out/'scores.json', rows)
    save_json(out/'manifest.json', {'protocol': protocol, 'sources': files, 'utc': datetime.now(timezone.utc).isoformat(),
                                  'data_attribution': 'Sportec/DFL, Bassek et al. (2025), Figshare28196177, CC BY4.0' if not args.synthetic else 'Synthetic generator',
                                  'weights_source': 'Random initialization; listed source partitions only',
                                  'artifact_sha256': {str(p.relative_to(out)): sha256(p) for p in out.glob('leave_out_*/*/*') if p.is_file()}})
    print('COMPLETE '+str(out), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--dfl-root', type=Path)
    source.add_argument('--synthetic', action='store_true')
    p.add_argument('--c1-run', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--held', nargs='+', choices=MATCHES, default=list(MATCHES))
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--hidden', type=int, default=22)
    p.add_argument('--chunk-samples', type=int, default=1024)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.threads < 1 or (not args.synthetic and args.c1_run is None) or (args.synthetic and args.c1_run is not None):
        p.error('Use positive threads and --c1-run only (and always) with external DFL')
    run(args)


if __name__ == '__main__':
    main()
