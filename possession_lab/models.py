from dataclasses import dataclass, asdict
import warnings
import numpy as np


def features(data):
    distances = np.linalg.norm(data.players - data.ball[:, None, :], axis=2)
    distances = np.where(np.isfinite(distances), distances, np.inf)
    near = np.column_stack([distances[:, data.teams == t].min(axis=1)
                            if np.any(data.teams == t) else np.full(len(data.ball), np.inf) for t in (0, 1)])
    # Match official slot-order tie breaking, not a hard-coded team priority.
    winner = data.teams[distances.argmin(axis=1)] if len(data.teams) else np.full(len(data.ball), 2)
    winner = np.where(np.isfinite(near).any(axis=1), winner, 2)
    ball_speed = np.full(len(data.ball), np.nan)
    player_speed = np.full(data.players.shape[:2], np.nan)
    ball_speed[1:] = np.linalg.norm(np.diff(data.ball, axis=0), axis=1) * data.fps
    player_speed[1:] = np.linalg.norm(np.diff(data.players, axis=0), axis=2) * data.fps
    local = np.where((distances <= 10) & np.isfinite(player_speed), player_speed, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median_speed = np.nanmedian(local, axis=1)
    stopped = (ball_speed < 0.3) & (median_speed < 1) & (np.isfinite(local).sum(axis=1) >= 4)
    outside = np.isfinite(data.ball).all(axis=1) & np.any((data.ball < 0) | (data.ball > np.asarray(data.pitch)), axis=1)
    return {'near': near, 'nearest': winner, 'ball_speed': ball_speed, 'stopped': stopped, 'outside': outside}


def baseline(feat, window):
    codes = feat['nearest']
    if window == 0:
        return codes.astype(np.int8).copy()
    n = len(codes)
    idx = np.arange(n)
    low, high = np.maximum(0, idx - window), np.minimum(n, idx + window + 1)
    votes = []
    for team in (0, 1):
        cs = np.r_[0, np.cumsum(codes == team)]
        votes.append(cs[high] - cs[low])
    home, away = votes
    return np.where(home + away == 0, 2, np.where(home >= away, 0, 1)).astype(np.int8)


@dataclass(frozen=True)
class RuleConfig:
    confirm_s: float = 0.2
    gap_s: float = 0.08
    dead_s: float = 0.0
    stopped_evidence: bool = False
    control_m: float = 2.0
    margin_m: float = 0.5
    restart_s: float = 0.2
    restart_speed: float = 1.0

    @property
    def name(self):
        return f'R_c{self.confirm_s:g}_g{self.gap_s:g}_d{self.dead_s:g}_s{int(self.stopped_evidence)}'


def rules(feat, config, fps=25):
    near = feat['near']
    n = len(near)
    labels = np.full(n, 3, dtype=np.int8)
    state = 3
    candidate, start, last = -1, -1, -100000
    dead_start = -1
    for i in range(n):
        d = near[i]
        best = int(np.argmin(d))
        # Exact equal distances are deliberately ambiguous in the rule model.
        controlled = np.isfinite(d[best]) and d[best] <= config.control_m and d[1-best] - d[best] >= config.margin_m
        dead_evidence = config.dead_s > 0 and (feat['outside'][i] or (config.stopped_evidence and feat['stopped'][i]))
        if dead_evidence:
            if dead_start < 0:
                dead_start = i
            if (i - dead_start + 1) / fps >= config.dead_s:
                state = 2
                labels[dead_start:i+1] = 2
                candidate = -1
        else:
            dead_start = -1
        if state == 2:
            eligible = controlled and feat['ball_speed'][i] > config.restart_speed and not dead_evidence
            duration = config.restart_s
        else:
            eligible = controlled and best != state and not dead_evidence
            duration = config.confirm_s
        if eligible:
            if candidate != best or (i-last)/fps > config.gap_s + 1e-9:
                candidate, start = best, i
            last = i
            if (i-start+1)/fps >= duration:
                state = best
                labels[start:i+1] = state
                candidate = -1
        elif candidate >= 0:
            # Contrary observed evidence cancels the candidate; missing data can bridge a short gap.
            if controlled or np.isfinite(d).any() or (i-last)/fps > config.gap_s + 1e-9:
                candidate = -1
        labels[i] = state
    return labels


def config_dict(config):
    return asdict(config)
