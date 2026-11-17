"""Apply the fixed BE gate to one half of aligned H/A/D emissions."""

import argparse
from pathlib import Path

import numpy as np

from .data import Tracking, sha256
from .external_paths import allowed_output


TRACKING_HZ = 25
STRIDE = 2
GAP_FRAMES = TRACKING_HZ
MIN_PLAYERS = 5
EXPERT_WEIGHT = 0.35
EMISSION_HZ = TRACKING_HZ / STRIDE
STAGE = 'pre-shared-decoder'
CLASSES = ('H', 'A', 'D')
EXPERTS = ('motion-a', 'deepsets', 'graph')
TRACKING_FIELDS = {'period', 'frame_ids', 'ball', 'players', 'teams', 'pitch', 'fps'}


def visibility_gate(data: Tracking):
    """Return eligibility, missing-run length and H/A counts on the emission grid.

    The run counts consecutive missing 25 Hz frame slots including the current
    one. A call starts with an empty history at the half boundary.
    """
    observed = np.isfinite(data.ball).all(axis=1)
    index = np.arange(len(observed))
    last_observed = np.maximum.accumulate(np.where(observed, index, -1))
    gap = (index - last_observed)[::STRIDE]
    visible = np.isfinite(data.players[::STRIDE]).all(axis=2)
    counts = np.column_stack([visible[:, data.teams == team].sum(axis=1) for team in (0, 1)])
    return (gap >= GAP_FRAMES) & (counts >= MIN_PLAYERS).all(axis=1), gap, counts


def _validate_emissions(p, n):
    p = np.asarray(p)
    if (p.shape != (n, 3) or p.dtype.kind != 'f' or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))
            or not np.allclose(p.sum(axis=1, dtype=np.float64), 1, atol=1e-7, rtol=0)):
        raise ValueError('Expected normalized floating H/A/D emission rows')
    return p


def mix_home_away(base, expert, eligible):
    """Return mixed emissions and the mask where both H/A conditionals exist."""
    eligible = np.asarray(eligible)
    if eligible.ndim != 1 or eligible.dtype.kind != 'b' or not len(eligible):
        raise ValueError('Expected a nonempty one-dimensional boolean gate')
    base = _validate_emissions(base, len(eligible))
    expert = _validate_emissions(expert, len(eligible))
    base_ha = base[:, :2].astype(np.float64)
    expert_ha = expert[:, :2].astype(np.float64)
    base_mass = base_ha.sum(axis=1)
    expert_mass = expert_ha.sum(axis=1)
    applied = eligible & (base_mass > 0) & (base[:, 2] < 1) & (expert_mass > 0)
    mixed = base.copy()
    base_cond = base_ha[applied] / base_mass[applied, None]
    expert_cond = expert_ha[applied] / expert_mass[applied, None]
    conditional = (1 - EXPERT_WEIGHT) * base_cond + EXPERT_WEIGHT * expert_cond
    mixed[applied, :2] = (1 - base[applied, 2].astype(np.float64))[:, None] * conditional
    return mixed, applied


def load_tracking(path):
    """Read the existing unlabelled Tracking NPZ without importing torch."""
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != TRACKING_FIELDS:
            raise ValueError('Expected the unlabelled Tracking NPZ fields')
        frames, teams, pitch = z['frame_ids'], z['teams'], z['pitch']
        ball, players = z['ball'], z['players']
        period, fps = z['period'].item(), z['fps'].item()
        if frames.ndim != 1 or frames.dtype.kind not in 'iu':
            raise ValueError('Expected integer frame IDs')
        if teams.ndim != 1 or teams.dtype.kind not in 'iu':
            raise ValueError('Expected integer H/A team codes')
        if (pitch.shape != (2,) or pitch.dtype.kind not in 'fi'
                or not np.isfinite(pitch).all() or (pitch <= 0).any()):
            raise ValueError('Expected two positive pitch dimensions')
        if (ball.dtype.kind not in 'fi' or players.dtype.kind not in 'fi'
                or np.isinf(ball).any() or np.isinf(players).any()):
            raise ValueError('Tracking coordinates must be numeric with NaN for missing values')
        if period not in ('firstHalf', 'secondHalf') or fps != TRACKING_HZ:
            raise ValueError('Expected one named half of 25 Hz tracking')
        return Tracking(period, frames, ball, players, teams, tuple(pitch), fps)


def load_emissions(path, match, data, tracking_hash):
    expected = {
        'match': match, 'half': data.period, 'fps': EMISSION_HZ,
        'probability_stage': STAGE, 'tracking_sha256': tracking_hash,
    }
    with np.load(path, allow_pickle=False) as z:
        required = set(expected) | {'frame', 'p_H', 'p_A', 'p_D'}
        if not required.issubset(z.files):
            raise ValueError(f'{path}: missing emission fields {sorted(required - set(z.files))}')
        for name, value in expected.items():
            if z[name].shape != () or z[name].item() != value:
                raise ValueError(f'{path}: mismatched {name}')
        frames = z['frame']
        if frames.dtype.kind not in 'iu' or not np.array_equal(frames, data.frame_ids[::STRIDE]):
            raise ValueError(f'{path}: frame IDs differ from the half-local emission grid')
        columns = [z[f'p_{label}'] for label in CLASSES]
        if any(col.shape != frames.shape or col.dtype.kind != 'f' for col in columns):
            raise ValueError(f'{path}: expected one floating value per class and frame')
        return _validate_emissions(np.column_stack(columns), len(frames))


def gate_arrays(data, match, base, experts, hashes):
    """Build the common pre-decoder archive from already aligned emissions."""
    eligible, gap, counts = visibility_gate(data)
    arrays = {'base': base}
    for name in EXPERTS:
        expert = experts[name]
        arm = 'base_plus_' + name.replace('-', '_')
        arrays[arm], arrays[arm + '_applied'] = mix_home_away(base, expert, eligible)
        arrays[arm + '_expert_ha_argmax'] = expert[:, :2].argmax(axis=1).astype(np.int8)
    return dict(
        arrays, gate_eligible=eligible, missing_run_frames_25hz=gap, visible_players_ha=counts,
        frame=data.frame_ids[::STRIDE], frame_ids_25hz=data.frame_ids, match=match, half=data.period,
        fps=EMISSION_HZ, classes=CLASSES, probability_stage=STAGE, input_names=list(hashes),
        input_sha256=list(hashes.values()), min_gap_frames_25hz=GAP_FRAMES,
        min_players_per_team=MIN_PLAYERS, expert_weight=EXPERT_WEIGHT,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', required=True)
    for name in ('tracking', 'base', *EXPERTS, 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if args.out.exists() or args.out.suffix.lower() != '.npz' or not allowed_output(args.out, repo):
        parser.error('Use a new .npz output under data/external/runs or outside repo')

    try:
        paths = {'tracking': args.tracking, 'base': args.base}
        paths.update({name: getattr(args, name.replace('-', '_')) for name in EXPERTS})
        hashes = {name: sha256(path) for name, path in paths.items()}
        tracking = load_tracking(args.tracking)
        base = load_emissions(args.base, args.match, tracking, hashes['tracking'])
        experts = {name: load_emissions(paths[name], args.match, tracking, hashes['tracking']) for name in EXPERTS}
        arrays = gate_arrays(tracking, args.match, base, experts, hashes)
        if any(sha256(path) != hashes[name] for name, path in paths.items()):
            raise ValueError('An input file changed during the run')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('xb') as output:
        np.savez_compressed(output, **arrays)
    eligible = arrays['gate_eligible']
    print(f'{args.match} {tracking.period}: {eligible.sum()}/{len(eligible)} eligible rows; {args.out}')


if __name__ == '__main__':
    main()
