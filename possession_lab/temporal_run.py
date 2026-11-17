"""C1: matched TCN/CRF comparison on external DFL, or a synthetic smoke test."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .data import Tracking, sha256
from .evaluation import official
from .external_paths import allowed_output
from .multimatch_run import MATCHES, RESERVED, SCENARIOS, partitions, fit_bundle, load_completed
from .probabilistic import NAMES, STRIDE, probability_metrics
from .robust import mask_ball
from .robust_run import read_match, score
from .run import save_json, save_table
from .temporal import TCNConfig, TemporalBundle, fit_tcn


METRICS = ('spell_f1', 'macro_f1', 'f1_D', 'takeover_25_f1', 'stoppage_25_f1',
           'nll', 'ece', 'brier', 'inference_seconds')


def synthetic_datasets():
    """Independent, intentionally easy toy matches; scores have no football meaning."""
    datasets = {}
    for index, match in enumerate(MATCHES):
        halves = []
        for half in range(2):
            rng = np.random.default_rng(700+index*10+half)
            n = 1801
            y = ((np.arange(n)//100+index+half) % 3).astype(np.int8)
            players = np.broadcast_to(np.array([[25., 30.], [30., 38.],
                                                 [75., 30.], [70., 38.]]), (n, 4, 2)).copy()
            players += rng.normal(0, .05, players.shape)
            ball = np.array([[25., 30.], [75., 30.], [-2., 20.]])[y]
            ball = ball+rng.normal(0, .03, ball.shape)
            tracking = Tracking(('firstHalf', 'secondHalf')[half],
                                np.arange(n)+half*10000, ball, players,
                                np.array([0, 0, 1, 1]), (105., 68.))
            halves.append((tracking, y))
        datasets[match] = halves
    return datasets


def save_prediction(path, match, data, probabilities, labels, *, probability_stage=None):
    """Compact per-frame contract for integration, on the original 25 Hz timeline."""
    p, y = np.asarray(probabilities), np.asarray(labels)
    n = len(data.frame_ids)
    if (p.shape != (n, 3) or y.shape != (n,) or not np.isfinite(p).all() or
            not np.issubdtype(y.dtype, np.integer) or
            (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-6) or
            not np.isin(y, [0, 1, 2]).all()):
        raise ValueError('Invalid prediction contract')
    coverage = np.isfinite(data.players).all(2).sum(1)/max(1, len(data.teams))
    metadata = {} if probability_stage is None else {'probability_stage': probability_stage}
    np.savez_compressed(path, schema='team-temporal-predictions-v1', match=match,
                        half=data.period, frame=data.frame_ids, p_H=p[:, 0], p_A=p[:, 1], p_D=p[:, 2],
                        label=np.array(['H', 'A', 'D'])[y], ball_available=np.isfinite(data.ball).all(1),
                        player_fraction=coverage, fps=data.fps, model_rate_hz=data.fps/STRIDE,
                        source='tracking', **metadata)


def summarize(rows, held):
    expected = {(m, v, scenario, seed) for m in held for v in ('crf', 'tcn')
                for scenario, _, seed, _ in SCENARIOS}
    keys = [(r['match'], r['variant'], r['scenario'], r['seed']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Incomplete or duplicate temporal evaluation grid')
    per_match = []
    for match in held:
        for scenario in dict.fromkeys(s[0] for s in SCENARIOS):
            for variant in ('crf', 'tcn'):
                values = [r for r in rows if (r['match'], r['scenario'], r['variant']) ==
                          (match, scenario, variant)]
                per_match.append(dict(match=match, scenario=scenario, variant=variant, seeds=len(values),
                                      **{k: float(np.mean([r[k] for r in values])) for k in METRICS}))
    paired = []
    for row in per_match:
        if row['variant'] != 'tcn':
            continue
        control = next(r for r in per_match if r['match'] == row['match'] and
                       r['scenario'] == row['scenario'] and r['variant'] == 'crf')
        paired.append(dict(match=row['match'], scenario=row['scenario'],
                           **{k+'_delta': row[k]-control[k] for k in METRICS}))
    return per_match, paired


def completed_case(path):
    """Resume only complete, unchanged prediction exports."""
    record = json.loads(path.read_text(encoding='utf-8'))
    for name, digest in record['prediction_sha256'].items():
        if Path(name).name != name or sha256(path.parent/name) != digest:
            raise ValueError('Prediction export changed: '+name)
    if len(record['prediction_sha256']) != 4 or len(record['rows']) != 2:
        raise ValueError('Incomplete temporal scenario')
    return record['rows']


def write_report(out, rows, held, synthetic, costs):
    per_match, paired = summarize(rows, held)
    save_table(out/'per_match.csv', per_match, ['match', 'scenario', 'variant', 'seeds', *METRICS])
    save_table(out/'paired_deltas.csv', paired, ['match', 'scenario', *[k+'_delta' for k in METRICS]])
    save_json(out/'costs.json', costs)
    lines = ['# C1: TCN versus CRF', '',
             'SYNTHETIC SMOKE TEST: no football accuracy claim.' if synthetic else
             f'{len(held)} of five external DFL development folds; complete held-out matches.',
             'Same 13 geometry features, 12.5 Hz, training/calibration masks, scaler, source split and decoder.',
             'Each half is scored independently; official counts are pooled before F1. Seeds are averaged within match.',
             'Final fixed TCN epoch, no model selection on the held-out match. No final refit or C4 claim.', '',
             '| Match | Scenario | CRF spellF1 | TCN spellF1 | TCN − CRF | TCN D-F1 |',
             '|---|---|---:|---:|---:|---:|']
    for delta in paired:
        match, scenario = delta['match'], delta['scenario']
        values = {r['variant']: r for r in per_match if r['match'] == match and r['scenario'] == scenario}
        lines.append(f"| {match} | {scenario} | {values['crf']['spell_f1']:.4f} | "
                     f"{values['tcn']['spell_f1']:.4f} | {delta['spell_f1_delta']:+.4f} | "
                     f"{values['tcn']['f1_D']:.4f} |")
    lines += ['', 'Mean of paired differences, weighting each held-out match equally:', '']
    for scenario in dict.fromkeys(s[0] for s in SCENARIOS):
        values = [r['spell_f1_delta'] for r in paired if r['scenario'] == scenario]
        lines.append(f'- {scenario}: {np.mean(values):+.4f} Spell F1 ({len(values)} matches).')
    lines += ['', 'Runtime and parameter counts: costs.json. Inference includes features, emissions, calibration and decoding;',
              'it excludes scoring, export and XML parsing. CPU/GPU device is recorded; unmeasured GPU cost is not estimated.',
              'TCN has finite convolutional context; CRF marginals use the complete half. Both share the subsequent chain decoder.',
              'Calibration tails share source matches with fitting; held-out matches are excluded from both.',
              'Synthetic masking does not reproduce all tracking failures. Overlapping folds are dependent.',
              'Reserved J03WQQ/J03WR9 and competition assets are not read. C2/C3 and independent reproduction remain pending.', '']
    (out/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    repo = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    held = list(args.held)
    if not held or len(set(held)) != len(held) or not set(held) <= set(MATCHES):
        raise ValueError('Choose unique DFL development matches')
    if not allowed_output(out, repo) or (out.exists() and not args.resume) or (args.resume and not out.is_dir()):
        raise ValueError('Use a new output under data/external/runs or outside repo; --resume requires an existing run')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    config = TCNConfig(hidden=args.hidden, epochs=args.epochs, chunk_samples=args.chunk_samples, seed=args.seed)
    code = {str(p.relative_to(repo)): sha256(p) for p in (repo/'possession_lab').glob('*.py')}
    packages = {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas', 'python-crfsuite', 'torch')}
    protocol = {'version': 'team-temporal-c1-v1', 'synthetic': args.synthetic, 'held': held,
                'hypothesis': 'A small TCN can improve held-out possession spells using the existing CRF features.',
                'tcn': asdict(config), 'features': list(NAMES), 'stride': STRIDE, 'crf_variant': 'augmentation',
                'partitions': 'multimatch_run.partitions: source first80/final20, 125-frame purge each side',
                'views': 'clean + missing40; identical match/half seeds from existing protocol',
                'calibration': 'Source tails only: emission temperature, chain(.25), chain temperature',
                'scenarios': SCENARIOS, 'reserved_unopened': list(RESERVED),
                'checkpoint_selection': 'Final fixed epoch; no early stopping or architecture selection',
                'code_sha256': code, 'scorer_sha256': sha256(repo/'evaluation/possession_metrics.py'),
                'dfl_root': None if args.synthetic else str(args.dfl_root.resolve()),
                'packages': packages, 'device': args.device, 'threads': args.threads,
                'platform': platform.platform()}
    protocol = json.loads(json.dumps(protocol))
    if args.resume:
        if json.loads((out/'protocol_before_run.json').read_text(encoding='utf-8')) != protocol:
            raise ValueError('Protocol, code, dependencies or input root changed; use a new run')
    else:
        out.mkdir(parents=True)
        save_json(out/'protocol_before_run.json', protocol)
    if args.synthetic:
        datasets, audits, files = synthetic_datasets(), {}, []
    else:
        datasets, audits, files = {}, {}, []
        for match in MATCHES:
            print('READ EXTERNAL DFL '+match, flush=True)
            datasets[match], audits[match], sources = read_match(args.dfl_root, match)
            files.extend(sources)
    if args.resume and json.loads((out/'inputs.json').read_text(encoding='utf-8')) != files:
        raise ValueError('Input hashes changed')
    save_json(out/'inputs.json', files)
    save_json(out/'audits.json', audits)
    pm, rows, costs = official(repo), [], []
    for match in held:
        sources = [m for m in MATCHES if m != match]
        train, cal, records = partitions(datasets, sources)
        fold = out/('leave_out_'+match)
        fold.mkdir(exist_ok=True)
        crf_path, tcn_path = fold/'crf', fold/'tcn'
        started = time.perf_counter()
        if crf_path.exists():
            crf = load_completed(crf_path)
        else:
            crf = fit_bundle(datasets, sources, 'augmentation', crf_path, pm)
            save_json(fold/'crf_cost.json', {'fit_and_calibration_seconds': time.perf_counter()-started})
        tcn = (TemporalBundle.load(tcn_path, args.device) if tcn_path.exists() else
               fit_tcn(train, cal, records, sources, tcn_path, config, args.device))
        if (crf.config['partitions'] != records or tcn.config['partitions'] != records or
                crf.config['training_matches'] != sources or tcn.config['training_matches'] != sources):
            raise ValueError('TCN/CRF split mismatch')
        np.testing.assert_array_equal(crf.model.scaler.mean, tcn.scaler.mean)
        np.testing.assert_array_equal(crf.model.scaler.scale, tcn.scaler.scale)
        np.testing.assert_array_equal(crf.config['transition'], tcn.config['transition'])
        np.testing.assert_array_equal(crf.config['prior'], tcn.config['prior'])
        crf_info = crf.model.tagger.info()
        costs.append({'match': match, 'tcn_parameters': tcn.config['parameters'],
                      'crf_coefficients': len(crf_info.state_features)+len(crf_info.transitions),
                      'tcn_fit_seconds': tcn.config['fit_seconds'],
                      'tcn_calibration_seconds': tcn.config['calibration_seconds'],
                      'crf': json.loads((fold/'crf_cost.json').read_text()), 'device': args.device})
        for scenario, fraction, seed, long in SCENARIOS:
            result = fold/f'{scenario}_{seed}.json'
            if result.exists():
                rows.extend(completed_case(result))
                continue
            cases = [(d if fraction is None else mask_ball(d, fraction, seed+10*MATCHES.index(match)+i, long)[0], y)
                     for i, (d, y) in enumerate(datasets[match])]
            batch, prediction_hashes = [], {}
            for name, bundle in (('crf', crf), ('tcn', tcn)):
                started = time.perf_counter()
                predictions = [bundle.predict(d) for d, _ in cases]
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                row = score(pm, cases, [y for _, y in predictions])
                prob = probability_metrics(np.concatenate([p for p, _ in predictions]),
                                           np.concatenate([y for _, y in cases]),
                                           np.concatenate([y for _, y in predictions]))
                row.update(match=match, variant=name, scenario=scenario, seed=seed,
                           inference_seconds=elapsed, **{k: prob[k] for k in ('nll', 'ece', 'brier')})
                for (d, _), (p, y) in zip(cases, predictions):
                    prediction_path = fold/f'{name}_{scenario}_{seed}_{d.period}.npz'
                    save_prediction(prediction_path, match, d, p, y)
                    prediction_hashes[prediction_path.name] = sha256(prediction_path)
                batch.append(row)
                print(f"EVAL {match} {scenario}/{seed} {name}: spellF1={row['spell_f1']:.4f}", flush=True)
            save_json(result, {'rows': batch, 'prediction_sha256': prediction_hashes})
            rows.extend(batch)
        save_json(out/'partial_scores.json', rows)
        write_report(out, rows, held[:held.index(match)+1], args.synthetic, costs)
    save_json(out/'scores.json', rows)
    save_json(out/'manifest.json', {'protocol': protocol, 'sources': files, 'utc': datetime.now(timezone.utc).isoformat(),
                                  'weights_source': 'Random initialization, trained only on listed source partitions',
                                  'data_source': 'Synthetic generator' if args.synthetic else
                                  'Sportec/DFL Figshare 28196177, Bassek et al. (2025), CC BY 4.0',
                                  'artifact_sha256': {str(p.relative_to(out)): sha256(p)
                                                      for p in out.glob('leave_out_*/*/*') if p.is_file()}})
    print('COMPLETE '+str(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dfl-root', type=Path)
    source.add_argument('--synthetic', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--held', nargs='+', choices=MATCHES, default=list(MATCHES))
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--chunk-samples', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    run(args)


if __name__ == '__main__':
    main()
