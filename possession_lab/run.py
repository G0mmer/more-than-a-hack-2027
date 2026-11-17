"""Run the first DFL development benchmark; never opens competition data."""
from .external_paths import allowed_output
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
from .data import load_dfl, sha256
from .models import features, baseline, rules, RuleConfig
from .evaluation import official, evaluate_period, aggregate
from .stress import perturb


def save_json(path, value):
    def convert(v):
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding='utf-8')


def save_table(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def score_model(pm, halves, feats, window=None, config=None):
    started = time.perf_counter()
    predictions = [baseline(f, window) if config is None else rules(f, config) for f in feats]
    inference_s = time.perf_counter() - started
    details = [evaluate_period(pm, gt, pred) for (_, gt), pred in zip(halves, predictions)]
    pooled = aggregate(pm, details)
    pooled['inference_s'] = inference_s
    return pooled, details, predictions


def error_rows(pm, halves, predictions, model):
    for (data, truth), pred in zip(halves, predictions):
        for run in pm.valid_runs(truth):
            filled = pm.fill_abstentions(pred, [run])
            pairs, missed, extra = pm.match_transitions(pm.transitions(truth, [run]), pm.transitions(filled, [run]), 25, 'pair')
            for status, items in [('missed', missed), ('spurious', extra)]:
                for frame, origin, target in items:
                    yield {'model': model, 'period': data.period, 'kind': status,
                           'native_frame': int(data.frame_ids[frame]), 'seconds_from_period_start': frame/25,
                           'from': 'HADX'[origin], 'to': 'HADX'[target]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='Only the public DFL J03WMX folder')
    parser.add_argument('--out', type=Path, required=True, help='New output folder under data/external/runs or outside the repository')
    parser.add_argument('--skip-stress', action='store_true', help='Debug only; report marks missing stress tests')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    if (not allowed_output(out, repo)) or out.exists():
        parser.error('--out must be a NEW folder under data/external/runs or outside the repository')
    source = args.data_dir.resolve()
    metas = list(source.glob('*matchinformation*J03WMX.xml'))
    positions = list(source.glob('*positions_raw_observed*J03WMX.xml'))
    if len(metas) != 1 or len(positions) != 1:
        parser.error('Expected exactly one metadata and one observed-positions XML for J03WMX')
    out.mkdir(parents=True)
    print('Auditing public DFL J03WMX...', flush=True)
    halves, audit = load_dfl(metas[0], positions[0])
    if audit['match_id'] != 'DFL-MAT-J03WMX':
        raise ValueError('First development benchmark is restricted to J03WMX; holdout matches are not opened')
    save_json(out/'audit.json', audit)
    if any(a['gt_coverage'] < 0.95 for a in audit['periods']):
        raise ValueError('Reference coverage below 95%; inspect audit before running model selection')
    pm = official(repo)
    feats = [features(d) for d, _ in halves]
    rows, details, configs, predictions = [], {}, {}, {}

    def execute(name, window=None, config=None):
        score, detail, pred = score_model(pm, halves, feats, window, config)
        score['model'] = name
        rows.append(score)
        details[name] = detail
        configs[name] = {'window': window} if config is None else asdict(config)
        predictions[name] = pred
        print(f"{name}: spellF1={score['spell_f1']:.4f}, macroF1={score['macro_f1']:.4f}", flush=True)

    for window in (0, 25, 50, 125):
        execute(f'B{window}', window=window)
    rule_configs = [RuleConfig(c, g) for c in (0.2, 0.4) for g in (0.08, 0.16)]
    for config in rule_configs:
        execute(config.name, config=config)
    # Development-only selection: never claim an independent test score.
    best_r = max(rule_configs, key=lambda c: next(r['spell_f1'] for r in rows if r['model'] == c.name))
    for dead_s in (0.4, 0.8):
        for stopped in (False, True):
            config = RuleConfig(best_r.confirm_s, best_r.gap_s, dead_s, stopped)
            execute(config.name, config=config)
    candidate = max((r for r in rows if r['model'].startswith('R_')), key=lambda r: r['spell_f1'])['model']
    candidate_config = RuleConfig(**configs[candidate])
    stress_rows = []
    if not args.skip_stress:
        for scenario in ('missing_0.2', 'missing_0.4', 'noise_0.3m', 'slots'):
            for seed in (0, 1, 2):
                changed, intervention = zip(*(perturb(d, scenario, seed) for d, _ in halves))
                stressed = [(d, gt) for d, (_, gt) in zip(changed, halves)]
                changed_feats = [features(d) for d in changed]
                for name in ('B50', candidate):
                    score, _, _ = score_model(pm, stressed, changed_feats, window=50 if name == 'B50' else None,
                                              config=None if name == 'B50' else candidate_config)
                    clean = next(r['spell_f1'] for r in rows if r['model'] == name)
                    score.update(model=name, scenario=scenario, seed=seed, delta_clean=score['spell_f1']-clean,
                                 masked_fraction=float(np.mean([x['masked_fraction'] for x in intervention])))
                    stress_rows.append(score)
            print('Stress completed: '+scenario, flush=True)
    fields = ['model', 'spell_f1', 'macro_f1', 'f1_D', 'pred_spells', 'gt_spells', 'graded_frames', 'abstain_frames',
              'takeover_12_f1', 'takeover_25_f1', 'stoppage_12_f1', 'stoppage_25_f1',
              'takeover_25_latency_s', 'stoppage_25_latency_s', 'inference_s']
    save_table(out/'scores.csv', rows, fields)
    save_table(out/'stress.csv', stress_rows, ['scenario', 'seed', 'masked_fraction', 'delta_clean']+fields)
    save_json(out/'scores.json', {'pooled': rows, 'per_half': details, 'configs': configs,
                               'candidate': candidate, 'role': 'development_only', 'stress': stress_rows})
    pred_dir = out/'predictions'
    pred_dir.mkdir()
    for name, per_half in predictions.items():
        for (data, _), pred in zip(halves, per_half):
            save_table(pred_dir/f'{name}_{data.period}.csv',
                       ({'frame': i, 'native_frame': int(data.frame_ids[i]), 'label': 'HADX'[int(code)]} for i, code in enumerate(pred)),
                       ['frame', 'native_frame', 'label'])
    save_table(out/'boundary_errors.csv',
               (row for name in ('B50', candidate) for row in error_rows(pm, halves, predictions[name], name)),
               ['model', 'period', 'kind', 'native_frame', 'seconds_from_period_start', 'from', 'to'])
    code_files = sorted((repo/'possession_lab').glob('*.py')) + [repo/'evaluation'/'possession_metrics.py', repo/'starter'/'baseline.py']
    manifest = {'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'platform': platform.platform(),
                'command_argv': sys.argv, 'seeds': [0, 1, 2], 'role': 'development_only',
                'sources': [{'path': str(p), 'size': p.stat().st_size, 'sha256': sha256(p)} for p in [metas[0], positions[0]]],
                'dataset_url': 'https://doi.org/10.6084/m9.figshare.28196177', 'dataset_license': 'CC-BY-4.0',
                'code_sha256': {str(p.relative_to(repo)): sha256(p) for p in code_files},
                'dependencies': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'python-dateutil', 'pytz', 'tzdata', 'six']}}
    save_json(out/'manifest.json', manifest)
    clean_best = next(r for r in rows if r['model'] == candidate)
    base = next(r for r in rows if r['model'] == 'B50')
    lines = ['# Pierwszy benchmark DFL — wynik rozwojowy', '',
             'J03WMX: jeden mecz, dwie połowy. Warianty wybrane na tym samym meczu; brak niezależnego testu.',
             'To zgodność z referencją DFL, nie prognoza wyniku konkursowego. Nie czytano danych konkursowych.', '',
             '| Model | spellF1 | macroF1 | F1 D | Posiadania pred/GT |', '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['spell_f1']:.4f} | {r['macro_f1']:.4f} | {r['f1_D']:.4f} | {r['pred_spells']}/{r['gt_spells']} |")
    lines += ['', f'Najlepszy wariant regułowy w tej iteracji: {candidate}. Różnica do B50: {clean_best["spell_f1"]-base["spell_f1"]:+.4f}.',
              '', '## Odporność', '', '| Scenariusz | B50, średnie spellF1 | Kandydat, średnie spellF1 |', '|---|---:|---:|']
    for scenario in sorted({r['scenario'] for r in stress_rows}):
        values = [np.mean([r['spell_f1'] for r in stress_rows if r['scenario']==scenario and r['model']==m]) for m in ('B50', candidate)]
        lines.append(f'| {scenario} | {values[0]:.4f} | {values[1]:.4f} |')
    if not stress_rows:
        lines.append('Testy odporności pominięte; nie wykonano pełnego etapu.')
    lines += ['', '## Ograniczenia i następny krok', '',
              '- DFL ma inny charakter błędów i referencji niż Metrica; sztuczne zaburzenia nie odtwarzają ATD.',
              '- Brak wideo i automatycznego Smart Tagging: ich wkład nie został zmierzony.',
              '- Progi reguł są hipotezami startowymi. Nie wykonano SSL, treningu sieci ani strojenia konkursowego.',
              '- GT X jest wykluczane przez oficjalną metrykę; braki wejść nie usuwają klatek.',
              '- Test slotów zmienia też tożsamość między próbkami, więc prędkości per slot mogą się psuć; to zamierzona próba odporności.',
              '- Brak przedziałów ufności dla transferu: jeden mecz nie pozwala ich wiarygodnie oszacować.', '',
              'Następny eksperyment: zamrozić wybrane konfiguracje i sprawdzić B50 oraz kandydata na drugim meczu rozwojowym J03WN1; dwóch meczów testowych jeszcze nie otwierać.', '',
              'Szczegóły: [audyt](audit.json), [wyniki](scores.csv), [odporność](stress.csv), [błędy granic](boundary_errors.csv), [manifest](manifest.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Report: '+str(out/'REPORT.md'), flush=True)


if __name__ == '__main__':
    main()
