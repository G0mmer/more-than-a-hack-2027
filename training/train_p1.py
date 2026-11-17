"""Step 5. Train the P1 team model (home/away refinement with mirrored-team augmentation).

Run from the repository root:
    python -B training/train_p1.py --inputs data/dfl_prepared --out runs/weights/p1
Recipe: masked_aug (motion_masked features with team-swap augmentation), eight epochs, seed 7.
The shipped weights were fitted on a Colab GPU; a CPU refit gives close but not identical weights.
"""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession_lab.final_round import datasets, get_model, verify  # noqa: E402
from possession_lab.multimatch_run import MATCHES, partitions  # noqa: E402


def main(a):
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    verify(a.inputs)
    train, cal, records = partitions(datasets(a.inputs), list(MATCHES))
    get_model(a.out, 'masked_aug', 7, train, cal, records, list(MATCHES), a.device, 8)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=Path('data/dfl_prepared'))
    p.add_argument('--out', type=Path, default=Path('runs/weights/p1'))
    p.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    main(p.parse_args())
