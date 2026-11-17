"""Input-only interventions: identical seeds and masks for every model."""
from dataclasses import replace
import numpy as np


def perturb(data, scenario, seed):
    rng = np.random.default_rng(seed)
    ball, players = data.ball.copy(), data.players.copy()
    mask = np.zeros(len(ball), dtype=bool)
    if scenario.startswith('missing'):
        target = int(len(ball) * float(scenario.split('_')[1]))
        # Sample block lengths then disjoint positions via union; never inspect labels.
        while mask.sum() < target:
            length = int(rng.integers(13, 51))
            start = int(rng.integers(0, len(ball)))
            mask[start:min(len(ball), start+length)] = True
        ball[mask] = np.nan
    elif scenario == 'noise_0.3m':
        ball += rng.normal(0, 0.3, ball.shape)
        players += rng.normal(0, 0.3, players.shape)
    elif scenario == 'slots':
        for start in range(0, len(ball), 25):
            stop = min(start+25, len(ball))
            for team in (0, 1):
                cols = np.flatnonzero(data.teams == team)
                players[start:stop, cols] = data.players[start:stop, rng.permutation(cols)]
    else:
        raise ValueError(scenario)
    return replace(data, ball=ball, players=players), {'masked_fraction': float(mask.mean())}
