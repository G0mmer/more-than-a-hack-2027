"""Step 3. Train the anchor TCN (H/A/D emissions from team geometry, 8-second context).

Run from the repository root:
    python -B training/train_anchor_tcn.py --inputs data/dfl_prepared --out runs/weights/anchor_tcn
Recipe: five DFL matches, first 80% of each half for fitting and the last 20% for calibration,
clean plus 40% synthetic ball gaps, eight epochs, seed 7, two CPU threads.
"""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession_lab.final_round import datasets, verify  # noqa: E402
from possession_lab.multimatch_run import MATCHES, partitions  # noqa: E402
from possession_lab.team_temporal import TeamConfig, fit  # noqa: E402
from possession_lab.team_temporal_run import prepared_views  # noqa: E402


def main(a):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    verify(a.inputs)
    train, cal, records = partitions(datasets(a.inputs), list(MATCHES))
    config = TeamConfig(context_seconds=8, representation='geometry', hidden=22, epochs=8,
                        chunk_samples=1024, learning_rate=.001, weight_decay=.001, seed=7)
    fit(*prepared_views(train, cal, False), list(MATCHES), records, a.out, config, 'cpu')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=Path('data/dfl_prepared'))
    p.add_argument('--out', type=Path, default=Path('runs/weights/anchor_tcn'))
    main(p.parse_args())
