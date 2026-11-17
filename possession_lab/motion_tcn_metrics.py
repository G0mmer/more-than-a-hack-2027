"""R2 diagnostics computed with the official one-to-one event matching."""
import numpy as np

from .evaluation import aggregate, counts_f1, evaluate_period


def close_contest(data, frame):
    if not np.isfinite(data.ball[frame]).all():
        return None
    distance = np.linalg.norm(data.players[frame] - data.ball[frame], axis=1)
    return all(np.any(distance[data.teams == team] <= 3.) for team in (0, 1))


def diagnostics(pm, originals, predictions):
    result = aggregate(pm, [evaluate_period(pm, y, pred)
                           for (_, y), pred in zip(originals, predictions)])
    counts = {name: 0 for name in ('short_gt', 'short_matched', 'false_splits',
                                 'false_merges', 'close_gt', 'close_matched',
                                 'close_unknown', 'restart_tp', 'restart_fn')}
    events = {tol: dict(tp=0, fp=0, fn=0, offsets=[]) for tol in (12, 25)}
    for (data, truth), pred in zip(originals, predictions):
        runs = pm.valid_runs(truth)
        gt_spells = pm.possession_spells(truth, runs)
        pred_spells = pm.possession_spells(pred, runs)
        overlaps = pm.spell_overlaps(gt_spells, pred_spells)
        matched = pm.matched_gt_flags(overlaps, len(gt_spells), .5)
        short = np.array([stop - start < 50 for start, stop, _ in gt_spells], dtype=bool)
        counts['short_gt'] += int(short.sum())
        counts['short_matched'] += int(matched[short].sum())
        # Count affected spells once. Each constituent overlap must last >=5 frames.
        gt_degree = np.zeros(len(gt_spells), dtype=int)
        pred_degree = np.zeros(len(pred_spells), dtype=int)
        for _, i, j in overlaps:
            a, b, _ = gt_spells[i]
            c, d, _ = pred_spells[j]
            if min(b, d) - max(a, c) >= 5:
                gt_degree[i] += 1
                pred_degree[j] += 1
        counts['false_splits'] += int((gt_degree >= 2).sum())
        counts['false_merges'] += int((pred_degree >= 2).sum())
        for run in runs:
            gt = pm.transitions(truth, [run])
            pr = pm.transitions(pred, [run])
            take_gt = [t for t in gt if t[1] < 2 and t[2] < 2]
            take_pr = [t for t in pr if t[1] < 2 and t[2] < 2]
            for tol, event in events.items():
                pairs, missed, extra = pm.match_transitions(take_gt, take_pr, tol, typed='pair')
                event['tp'] += len(pairs)
                event['fp'] += len(extra)
                event['fn'] += len(missed)
                event['offsets'].extend((p[0] - g[0]) / 25 for g, p in pairs)
                if tol == 25:
                    found = {g[0] for g, _ in pairs}
                    for g in take_gt:
                        close = close_contest(data, g[0])
                        counts['close_unknown'] += int(close is None)
                        if close:
                            counts['close_gt'] += 1
                            counts['close_matched'] += int(g[0] in found)
            restart_gt = [(i, 1, 0) for i, a, b in gt if a == 2 and b < 2]
            restart_pr = [(i, 1, 0) for i, a, b in pr if a == 2 and b < 2]
            pairs, missed, _ = pm.match_transitions(restart_gt, restart_pr, 25, typed='pair')
            counts['restart_tp'] += len(pairs)
            counts['restart_fn'] += len(missed)
    result.update(counts)
    result['short_recall'] = counts['short_matched'] / counts['short_gt'] if counts['short_gt'] else None
    result['close_recall'] = counts['close_matched'] / counts['close_gt'] if counts['close_gt'] else None
    total = counts['restart_tp'] + counts['restart_fn']
    result['restart_recall'] = counts['restart_tp'] / total if total else 1.
    for tol, event in events.items():
        tp, fp, fn = (event[k] for k in ('tp', 'fp', 'fn'))
        prefix = f'takeover_{tol}'
        result.update({f'{prefix}_{k}': event[k] for k in ('tp', 'fp', 'fn')})
        result[prefix + '_recall'] = tp / (tp + fn) if tp + fn else None
        result[prefix + '_precision'] = tp / (tp + fp) if tp + fp else None
        result[prefix + '_f1'] = counts_f1(tp, fp, fn)
        result[prefix + '_mae_s'] = float(np.mean(np.abs(event['offsets']))) if event['offsets'] else None
    return result
