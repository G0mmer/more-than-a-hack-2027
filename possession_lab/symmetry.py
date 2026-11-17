"""Opt-in C2 inference symmetry; never modifies the frozen bundle."""
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from .probabilistic import STRIDE, temperature
from .probabilistic_run import fusion
from .team_temporal import prepare


@dataclass(frozen=True)
class Transform:
    x: bool = False
    y: bool = False
    swap: bool = False

    @property
    def name(self):
        return '_'.join(k for k, v in zip(('x', 'y', 'swap'), (self.x, self.y, self.swap)) if v) or 'identity'

    def compose(self, other):
        return Transform(self.x ^ other.x, self.y ^ other.y, self.swap ^ other.swap)

    def tracking(self, data):
        # Promote before subtraction: repeated float32 reflections can move points
        # across discrete feature thresholds. No clipping or missing-value fill.
        ball = np.array(data.ball, dtype=np.float64, copy=True)
        players = np.array(data.players, dtype=np.float64, copy=True)
        for axis, enabled in enumerate((self.x, self.y)):
            if enabled:
                for array in (ball, players):
                    finite = np.isfinite(array[..., axis])
                    array[..., axis][finite] = data.pitch[axis] - array[..., axis][finite]
        return replace(data, ball=ball, players=players, frame_ids=data.frame_ids.copy(),
                       teams=1-data.teams if self.swap else data.teams.copy())

    def columns(self, p):
        p = np.asarray(p)
        if p.shape[-1] != 3:
            raise ValueError('Expected H/A/D columns')
        return p[..., [1, 0, 2] if self.swap else [0, 1, 2]].copy()

    def labels(self, y):
        y = np.asarray(y)
        if not np.isin(y, [0, 1, 2]).all():
            raise ValueError('Expected H/A/D labels')
        return np.where(y < 2, 1-y, y) if self.swap else y.copy()


GROUP = tuple(Transform(*flags) for flags in product((False, True), repeat=3))


def symmetric_decoder(transition, prior):
    transition, prior = np.asarray(transition, dtype=float), np.asarray(prior, dtype=float)
    if (transition.shape != (3, 3) or prior.shape != (3,) or
            not np.isfinite(transition).all() or not np.isfinite(prior).all() or
            (transition <= 0).any() or (prior <= 0).any() or
            not np.allclose(transition.sum(1), 1) or not np.isclose(prior.sum(), 1)):
        raise ValueError('Expected positive normalized H/A/D decoder')
    order = [1, 0, 2]
    return ((transition + transition[np.ix_(order, order)])/2,
            (prior + prior[order])/2)


def emissions(bundle, data):
    seq = prepare(data, bundle.config['model']['representation'] != 'geometry',
                  feature_set=bundle.config['model'].get('feature_set', 'geometry'))
    # raw is already a probability, not a logit.
    return temperature(bundle.raw(seq), bundle.config['emission_temperature'])


class SymmetryAdapter:
    """Average eight calibrated emissions, then decode once per half.

    Viterbi keeps the baseline first-index argmax tie rule. At exact H/A
    ties the chosen path need not be equivariant; marginals still are.
    """
    def __init__(self, bundle):
        self.bundle = bundle
        self.config = deepcopy(bundle.config)
        self.transition, self.prior = symmetric_decoder(self.config['transition'], self.config['prior'])
        self.config.update(variant='symmetry-eight-v1', transforms=[g.name for g in GROUP],
                           transition=self.transition.tolist(), prior=self.prior.tolist(),
                           combination='mean calibrated emissions before one decoder',
                           tie_rule='baseline first-index argmax; H/A ties may disagree')

    def emissions(self, data):
        # Sort each scalar orbit to make summation independent of group order.
        orbit = np.stack([g.columns(emissions(self.bundle, g.tracking(data))) for g in GROUP])
        return np.sort(orbit, axis=0).mean(axis=0)

    def decode(self, p, frames):
        q, y = fusion([p], [1.], self.transition, self.prior, .25)
        q = temperature(q, self.config['chain_temperature'])
        return np.repeat(q, STRIDE, axis=0)[:frames], np.repeat(y, STRIDE)[:frames]

    def predict(self, data):
        return self.decode(self.emissions(data), len(data.ball))


def audit(bundle, data, adapter=False):
    model = SymmetryAdapter(bundle) if adapter else bundle
    get_emissions = model.emissions if adapter else lambda d: emissions(bundle, d)
    def decode(p, frames):
        if adapter:
            return model.decode(p, frames)
        q, y = fusion([p], [1.], np.asarray(bundle.config['transition']), np.asarray(bundle.config['prior']), .25)
        return (np.repeat(temperature(q, bundle.config['chain_temperature']), STRIDE, axis=0)[:frames],
                np.repeat(y, STRIDE)[:frames])
    reference = get_emissions(data)
    q, y = decode(reference, len(data.ball))
    rows = []
    for g in GROUP:
        changed = g.tracking(data)
        raw = get_emissions(changed)
        p = g.columns(raw)
        qq, yy = decode(raw, len(data.ball))
        qq, yy = g.columns(qq), g.labels(yy)
        rows.append(dict(transform=g.name, emission_max=float(np.max(np.abs(reference-p))),
                         marginal_max=float(np.max(np.abs(q-qq))),
                         label_disagreements=int(np.sum(y != yy)),
                         dead_disagreements=int(np.sum((y == 2) != (yy == 2))), frames=len(y)))
    return rows
