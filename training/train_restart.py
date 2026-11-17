"""Step 7. Train the dead-ball state and restart heads (two L2 logistic regressions).

Run from the repository root:
    python -B training/train_restart.py --inputs data/dfl_prepared --out runs/weights/restart.json
Recipe: player-motion evidence at 5 Hz from clean and ball-removed noisy tracking, class-balanced
sampling with seed 711578, L2 0.012, all five DFL matches.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession_lab.final_round import datasets, verify  # noqa: E402
from possession_lab.multimatch_run import MATCHES  # noqa: E402
from possession_lab.restart_supervision import fit_heads, strong_targets, tracking_features  # noqa: E402
from possession_lab.stoppage_evidence import STEP, corrupt, evidence, player_evidence  # noqa: E402


def bank(data):
    prepared = {}
    for match, periods in data.items():
        rows = []
        for i, (original, truth) in enumerate(periods):
            seed = 1000+10*MATCHES.index(match)+i
            clean = player_evidence(original)
            noisy = player_evidence(corrupt(original, 'no_ball_noise', seed))
            for scenario, players in (('clean', clean), ('no_ball_noise', noisy)):
                f = {k: v.astype(np.float32) for k, v in
                     evidence(corrupt(original, scenario, seed), players).items()}
                rows.append((tracking_features(f), strong_targets(truth[::STEP]), f['mv'] > .5))
        prepared[match] = rows
    return prepared


def main(a):
    verify(a.inputs)
    heads = fit_heads(bank(datasets(a.inputs)), list(MATCHES), None, None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({'dfl_restart': heads}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=Path('data/dfl_prepared'))
    p.add_argument('--out', type=Path, default=Path('runs/weights/restart.json'))
    main(p.parse_args())
