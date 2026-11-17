"""Step 6. Train the player-movement H/A expert used when the ball is not observed.

Run from the repository root:
    python -B training/train_team_motion.py --inputs data/dfl_prepared --out runs/weights/team_motion.pt
Recipe: player features only, clean coordinates paired with 0.3 m noise (seed 1000+10*match+half),
six epochs, seed 7, two CPU threads.
"""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession_lab.final_round import datasets, verify  # noqa: E402
from possession_lab.multimatch_run import MATCHES  # noqa: E402
from possession_lab.stoppage_evidence import treatment  # noqa: E402
from possession_lab.team_identity import fit_specialists, player_features  # noqa: E402
from possession_lab.team_shape import STRIDE  # noqa: E402


def main(a):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    verify(a.inputs)
    if a.out.exists():
        raise ValueError('Use a new output path')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    data = datasets(a.inputs)
    halves = []
    for j, match in enumerate(MATCHES):
        for i, (tracking, truth) in enumerate(data[match]):
            noisy = treatment(tracking, 'noise', 1000+10*j+i)
            halves.append((player_features(tracking), player_features(noisy), truth[::STRIDE]))
    fit_specialists(halves, a.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=Path('data/dfl_prepared'))
    p.add_argument('--out', type=Path, default=Path('runs/weights/team_motion.pt'))
    main(p.parse_args())
