"""R2 paired training; one random raw-tracking transform per sequence/epoch.

The scaler and original decoder statistics remain identical to the control.
Decoder symmetrization is a separate inference ablation, not a training change.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from .baseline_suite import initialize
from .data import sha256
from .multimatch_run import MATCHES, RESERVED, partitions
from .robust import mask_ball
from .robust_run import read_match
from .run import save_json
from .symmetry import GROUP
from .team_temporal import TeamConfig, prepare, fit
from .team_temporal_run import prepared_views


class RawSymmetryEpochs:
    """Separate RNG never changes baseline chunk order or model initialization."""
    def __init__(self, training, seed=7, feature_set='geometry', player_sets=False):
        self.views = [(d, y) for d, y, _ in training] + [
            (mask_ball(d, .4, seed_)[0], y) for d, y, seed_ in training]
        self.seed = seed
        self.feature_set, self.player_sets = feature_set, player_sets
        self.history = []

    def __call__(self, epoch):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 20261112]))
        indices = rng.integers(len(GROUP), size=len(self.views))
        self.history.append(dict(epoch=epoch, transforms=[GROUP[i].name for i in indices]))
        # Transform the whole continuous partition, including every chunk's halo.
        # Derivatives are computed afterwards, never across independent transforms.
        return [prepare(GROUP[i].tracking(d), self.player_sets, GROUP[i].labels(y), self.feature_set)
                for i, (d, y) in zip(indices, self.views)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dfl-root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--held', nargs='+', choices=MATCHES, default=list(MATCHES))
    args = parser.parse_args()
    if len(set(args.held)) != len(args.held):
        parser.error('Duplicate held-out match')
    if not args.synthetic and args.dfl_root is None:
        parser.error('--dfl-root is required for real data')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = TeamConfig()
    initialize(args.out, 'r2-symmetry-train', dict(
        recipe=asdict(config), synthetic=args.synthetic, held=args.held,
        reserved_unopened=list(RESERVED), variants=['tcn', 'augmented'],
        p1_commit='36b87715467141c9d48821a8a07986ad0a109d0a',
        augmentation='uniform eight transforms; one per whole source sequence per epoch; before features',
        controls='same source-only scaler, native prior/transitions, calibration, steps and chunk order',
        split='whole-match OOF; source80/calibration20; purge125 each side',
        intended_evaluation=['tcn', 'tcn_symmetric_decoder', 'p1', 'augmented', 'augmented_symmetric_decoder'],
        football_quality_claim=False))
    if args.synthetic:
        from .temporal_run import synthetic_datasets
        datasets = synthetic_datasets()
    else:
        datasets = {}
        for match in MATCHES:
            print('READ EXTERNAL '+match, flush=True)
            datasets[match], audit, files = read_match(args.dfl_root, match)
            save_json(args.out/(match+'_inputs.json'), dict(audit=audit, files=files))
    for held in args.held:
        sources = [m for m in MATCHES if m != held]
        training, calibration, records = partitions(datasets, sources)
        prepared, calibrated, clean = prepared_views(training, calibration, False)
        for variant in ('tcn', 'augmented'):
            folder = args.out/('leave_out_'+held)/variant
            augment = RawSymmetryEpochs(training, config.seed) if variant == 'augmented' else None
            fit(prepared, calibrated, clean, sources, records, folder, config,
                epoch_training=augment)
            save_json(folder/'augmentation.json', dict(
                variant=variant, schedule=[] if augment is None else augment.history,
                scaler='unchanged source-only original views',
                decoder='native; symmetrization reserved for separate evaluation'))
    save_json(args.out/'complete.json', dict(synthetic=args.synthetic, held=args.held,
        model_files={str(p.relative_to(args.out)): sha256(p) for p in sorted(args.out.rglob('model.pt'))},
        status='training complete; quality and symmetry evaluation still required'))


if __name__ == '__main__':
    main()
