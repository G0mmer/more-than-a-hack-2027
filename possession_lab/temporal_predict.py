"""Run a frozen TCN on one explicit, unlabelled Tracking NPZ (one half)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import Tracking
from .external_paths import allowed_output
from .temporal import TemporalBundle
from .temporal_run import save_prediction


def load_tracking(path):
    with np.load(path, allow_pickle=False) as z:
        expected = {'period', 'frame_ids', 'ball', 'players', 'teams', 'pitch', 'fps'}
        if set(z.files) != expected:
            raise ValueError('Supply only the documented unlabelled Tracking fields')
        frames, pitch = z['frame_ids'], z['pitch']
        if (frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer) or
                pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any()):
            raise ValueError('Integer frame IDs and two finite positive pitch dimensions required')
        if z['teams'].ndim != 1 or not np.issubdtype(z['teams'].dtype, np.integer):
            raise ValueError('Integer team codes required')
        if np.isinf(z['ball']).any() or np.isinf(z['players']).any():
            raise ValueError('Missing coordinates must be NaN, not infinity')
        return Tracking(str(z['period'].item()), frames.copy(), z['ball'].copy(),
                        z['players'].copy(), z['teams'].copy(), tuple(z['pitch']), float(z['fps']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tracking', type=Path, required=True)
    parser.add_argument('--match', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.out.exists() or args.out.suffix != '.npz' or not allowed_output(args.out, repo):
        parser.error('Use a new .npz output under data/external/runs or outside repo')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = load_tracking(args.tracking)
    schema = json.loads((args.model/'bundle.json').read_text())['schema']
    if schema == 'team-temporal-c3-v1':
        from .transformer_temporal import C3Bundle
        bundle = C3Bundle.load(args.model)
    elif schema == 'team-temporal-c2-v1':
        from .team_temporal import TeamTemporalBundle
        bundle = TeamTemporalBundle.load(args.model)
    elif schema == 'team-temporal-c1-v1':
        bundle = TemporalBundle.load(args.model)
    else:
        parser.error('Unsupported temporal checkpoint schema')
    probabilities, labels = bundle.predict(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_prediction(args.out, args.match, data, probabilities, labels)


if __name__ == '__main__':
    main()
