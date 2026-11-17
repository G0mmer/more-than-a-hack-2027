"""Pool counts from the unchanged official metric, without joining halves."""
import importlib.util
from pathlib import Path
import numpy as np


def official(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    path = root / 'evaluation' / 'possession_metrics.py'
    spec = importlib.util.spec_from_file_location('official_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def counts_f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else 0.0


def restart_counts(pm, truth, prediction):
    """Matched and missed D-to-alive boundaries at the baseline's +/- 25 frames."""
    tp = fn = 0
    for run in pm.valid_runs(truth):
        gt = [(i, 1, 0) for i, a, b in pm.transitions(truth, [run]) if a == 2 and b < 2]
        pred = [(i, 1, 0) for i, a, b in pm.transitions(prediction, [run]) if a == 2 and b < 2]
        pairs, missed, _ = pm.match_transitions(gt, pred, 25, typed='pair')
        tp += len(pairs)
        fn += len(missed)
    return tp, fn


def evaluate_period(pm, truth, prediction):
    if len(truth) != len(prediction) or not np.isin(prediction, [0, 1, 2, 3]).all():
        raise ValueError('Invalid prediction')
    result = pm.evaluate(truth, prediction, bootstrap_n=0)
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(prediction, runs)
    groups = {}
    for tol in (12, 25):
        for kind in ('takeover', 'stoppage'):
            tp = fp = fn = 0
            offsets = []
            for run in runs:
                gt = pm.transitions(truth, [run])
                pred = pm.transitions(filled, [run])
                if kind == 'takeover':
                    gt = [t for t in gt if t[1] < 2 and t[2] < 2]
                    pred = [t for t in pred if t[1] < 2 and t[2] < 2]
                else:
                    # Binary alive/dead: restart team is evaluated separately by H/A metrics.
                    gt = [(i, int(a == 2), int(b == 2)) for i, a, b in gt if (a == 2) != (b == 2)]
                    pred = [(i, int(a == 2), int(b == 2)) for i, a, b in pred if (a == 2) != (b == 2)]
                pairs, missed, extra = pm.match_transitions(gt, pred, tol, typed='pair')
                tp += len(pairs)
                fp += len(extra)
                fn += len(missed)
                offsets.extend((p[0]-g[0])/25 for g, p in pairs)
            groups[f'{kind}_{tol}'] = {'tp': tp, 'fp': fp, 'fn': fn, 'latency_s': offsets}
    result['boundary_groups'] = groups
    return result


def aggregate(pm, results):
    cm = np.sum([np.asarray(r['confusion']) for r in results], axis=0)
    cm3 = pm.scored_confusion(cm)
    spells = {}
    for iou in pm.SPELL_IOUS:
        tp = sum(r['spell_f1'][iou]['n_matched'] for r in results)
        fp = sum(r['spell_f1'][iou]['n_spurious'] for r in results)
        fn = sum(r['spell_f1'][iou]['n_missed'] for r in results)
        spells[str(iou)] = {'tp': tp, 'fp': fp, 'fn': fn, 'f1': counts_f1(tp, fp, fn)}
    out = {'spell_f1': float(np.mean([s['f1'] for s in spells.values()])),
           'macro_f1': pm.macro_f1(cm3), 'f1_D': pm.per_class_prf(cm3, 2)[2],
           'pred_spells': sum(r['n_pred_spells'] for r in results),
           'gt_spells': sum(r['n_gt_spells'] for r in results),
           'graded_frames': int(cm3.sum()), 'abstain_frames': int(cm3[:, 3].sum()),
           'confusion': cm.tolist(), 'spell_counts': spells}
    for key in results[0]['boundary_groups']:
        tp = sum(r['boundary_groups'][key]['tp'] for r in results)
        fp = sum(r['boundary_groups'][key]['fp'] for r in results)
        fn = sum(r['boundary_groups'][key]['fn'] for r in results)
        lag = [v for r in results for v in r['boundary_groups'][key]['latency_s']]
        out[key + '_f1'] = counts_f1(tp, fp, fn)
        out[key + '_latency_s'] = float(np.mean(lag)) if lag else None
    return out
