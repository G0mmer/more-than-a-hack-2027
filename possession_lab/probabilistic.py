"""Geometry-only fuzzy, supervised Gaussian HMM and linear-chain CRF models."""
from dataclasses import dataclass
import json
from pathlib import Path
import warnings
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax, expit
import pycrfsuite

STRIDE = 2  # 12.5 Hz inference; repeat each result back to the 25 Hz scoring grid.
NAMES = ("dist_H", "dist_A", "margin", "ball_speed", "edge", "ball_x", "ball_y",
         "ball_available", "nearby_count", "nearby_speed", "player_count", "close_H", "close_A")


def geometry(data, mask_motion=False):
    ball = data.ball.astype(float)
    players = data.players.astype(float)
    d = np.linalg.norm(players-ball[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    near = np.column_stack([d[:, data.teams == t].min(axis=1) for t in (0, 1)])
    speed = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1)*25]
    ps = np.vstack([np.full((1, players.shape[1]), np.nan),
                    np.linalg.norm(np.diff(players, axis=0), axis=2)*25])
    if mask_motion:
        speed = np.where(speed <= 60., speed, np.nan)
        ps = np.where(ps <= 15., ps, np.nan)
    local = np.where(d <= 10, ps, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        motion = np.nanmedian(local, axis=1)
    edge = np.minimum.reduce([ball[:, 0], data.pitch[0]-ball[:, 0], ball[:, 1], data.pitch[1]-ball[:, 1]])
    nh, na = np.minimum(near[:, 0], 10), np.minimum(near[:, 1], 10)
    x = np.column_stack([nh/10, na/10, (na-nh)/10,
        np.log1p(np.clip(speed, 0, 40))/np.log(41), np.clip(edge/5, -1, 1),
        ball[:, 0]/data.pitch[0], ball[:, 1]/data.pitch[1],
        np.isfinite(ball).all(axis=1), np.minimum(np.sum(d <= 10, axis=1), 10)/10,
        np.log1p(np.clip(motion, 0, 15))/np.log(16),
        np.isfinite(players).all(axis=2).sum(axis=1)/22,
        np.minimum(np.sum(d[:, data.teams == 0] <= 3, axis=1), 4)/4,
        np.minimum(np.sum(d[:, data.teams == 1] <= 3, axis=1), 4)/4])
    return np.nan_to_num(x, nan=0, posinf=1, neginf=-1).astype(np.float64)


def split_ranges(n, purge=125):
    a, b = int(n*.6), int(n*.8)
    if a <= purge or b-a <= 2*purge or n-b <= purge:
        raise ValueError("Sequence too short for chronological split")
    return {"fit": (0, a-purge), "select": (a+purge, b-purge),
            "refit": (0, b-purge), "calibrate": (b+purge, n)}


@dataclass
class Sequence:
    key: str
    x: np.ndarray
    y: np.ndarray
    full_y: np.ndarray
    start: int
    stop: int


def make_sequence(key, x, y, start=0, stop=None):
    stop = len(y) if stop is None else stop
    xx, yy = x[start:stop], y[start:stop]
    if len(xx) != len(yy) or not np.isin(yy, [0, 1, 2]).all():
        raise ValueError("The first learned benchmark requires complete H/A/D reference")
    return Sequence(key, xx[::STRIDE].copy(), yy[::STRIDE].copy(), yy.copy(), start, stop)


def expand(array, sequence):
    return np.repeat(array, STRIDE, axis=0)[:len(sequence.full_y)]


class Standardizer:
    def fit(self, sequences):
        x = np.concatenate([s.x for s in sequences])
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 0.05)
        return self
    def transform(self, x):
        return np.clip((x-self.mean)/self.scale, -10, 10)


def transition_parameters(sequences):
    counts = np.full((3, 3), 0.1)
    prior = np.ones(3)
    for s in sequences:
        np.add.at(counts, (s.y[:-1], s.y[1:]), 1)
        prior += np.bincount(s.y, minlength=3)
    return counts/counts.sum(axis=1, keepdims=True), prior/prior.sum()


def infer_chain(log_emission, transition, prior):
    """Scaled forward/backward marginals and exact Viterbi path for three states."""
    n = len(log_emission)
    emission = np.exp(log_emission-log_emission.max(axis=1, keepdims=True))
    emission = np.maximum(emission, 1e-250)
    alpha = np.empty_like(emission)
    scale = np.empty(n)
    alpha[0] = prior*emission[0]
    scale[0] = alpha[0].sum()
    alpha[0] /= scale[0]
    log_t = np.log(np.maximum(transition, 1e-250))
    delta = np.log(np.maximum(prior, 1e-250))+log_emission[0]
    back = np.zeros((n, 3), dtype=np.int8)
    for i in range(1, n):
        alpha[i] = (alpha[i-1]@transition)*emission[i]
        scale[i] = alpha[i].sum()
        alpha[i] /= scale[i]
        options = delta[:, None]+log_t
        back[i] = options.argmax(axis=0)
        delta = options.max(axis=0)+log_emission[i]
        delta -= delta.max()
    posterior = np.empty_like(alpha)
    beta = np.ones(3)
    posterior[-1] = alpha[-1]
    for i in range(n-2, -1, -1):
        beta = transition@(emission[i+1]*beta)/scale[i+1]
        # Arbitrary beta scaling is safe once the posterior is renormalized.
        beta /= max(beta.max(), 1e-250)
        posterior[i] = alpha[i]*beta
        posterior[i] /= posterior[i].sum()
    path = np.empty(n, dtype=np.int8)
    path[-1] = delta.argmax()
    for i in range(n-2, -1, -1):
        path[i] = back[i+1, path[i+1]]
    return posterior, path


class GaussianHMM:
    def __init__(self, weight=1.0, floor=0.05):
        self.weight, self.floor = weight, floor
    def fit(self, sequences, destination):
        self.scaler = Standardizer().fit(sequences)
        x = np.concatenate([self.scaler.transform(s.x) for s in sequences])
        y = np.concatenate([s.y for s in sequences])
        if set(y) != {0, 1, 2}:
            raise ValueError("Training set lacks a state")
        self.means = np.array([x[y == k].mean(axis=0) for k in range(3)])
        self.variances = np.array([np.maximum(x[y == k].var(axis=0), self.floor) for k in range(3)])
        self.transition, self.prior = transition_parameters(sequences)
        np.savez(destination.with_suffix(".npz"), means=self.means, variances=self.variances,
                 transition=self.transition, prior=self.prior, mean=self.scaler.mean, scale=self.scaler.scale,
                 emission_weight=self.weight, variance_floor=self.floor)
        return self
    def predict(self, sequence):
        x = self.scaler.transform(sequence.x)
        log_e = -.5*np.sum(np.log(2*np.pi*self.variances)[None, :, :]
                          +(x[:, None, :]-self.means[None, :, :])**2/self.variances[None, :, :], axis=2)
        return infer_chain(log_e*self.weight, self.transition, self.prior)


class Fuzzy:
    def __init__(self, radius=2.0, memory=0.2):
        self.radius, self.memory = radius, memory
    def fit(self, sequences, destination):
        destination.with_suffix(".json").write_text(json.dumps({"radius": self.radius, "memory_s": self.memory}), encoding="utf-8")
        return self
    def predict(self, sequence):
        x = sequence.x
        dh, da = x[:, 0]*10, x[:, 1]*10
        speed = np.expm1(x[:, 3]*np.log(41))
        local_speed = np.expm1(x[:, 9]*np.log(16))
        out = expit(-x[:, 4]*5/.2)
        still = expit((.3-speed)/.15)*expit((1-local_speed)/.4)*expit((x[:, 8]*10-3.5))
        dead = np.clip(1-(1-out)*(1-.95*still), .01, .99)
        team_h = expit((da-dh)/self.radius)
        certainty = np.exp(-np.minimum(dh, da)/5)
        team_h = .5+(team_h-.5)*certainty
        probability = np.column_stack([(1-dead)*team_h, (1-dead)*(1-team_h), dead])
        mix = 1 if self.memory == 0 else 1-np.exp(-(STRIDE/25)/self.memory)
        for i in range(1, len(x)):
            if x[i, 7] < .5:
                probability[i] = probability[i-1]
            else:
                probability[i] = mix*probability[i]+(1-mix)*probability[i-1]
        probability = normalize(probability)
        return probability, probability.argmax(axis=1).astype(np.int8)


def crf_items(x, context):
    offsets = [0] if context == 0 else [-context, 0, context]
    n = len(x)
    blocks = [x[np.clip(np.arange(n)+offset, 0, n-1)] for offset in offsets]
    values = np.column_stack([np.ones(n)]+blocks)
    names = ["bias"]+[f"{offset}:{name}" for offset in offsets for name in NAMES]
    return pycrfsuite.ItemSequence([dict(zip(names, map(float, row))) for row in values])


class LinearCRF:
    def __init__(self, c2=1.0, context=0):
        self.c2, self.context = c2, context
    def fit(self, sequences, destination):
        self.scaler = Standardizer().fit(sequences)
        trainer = pycrfsuite.Trainer(verbose=False)
        for sequence in sequences:
            trainer.append(crf_items(self.scaler.transform(sequence.x), self.context), [str(int(y)) for y in sequence.y])
        trainer.set_params({"c1": 0, "c2": self.c2, "max_iterations": 80,
                            "feature.possible_transitions": True, "feature.possible_states": True})
        self.path = destination.with_suffix(".crfsuite")
        trainer.train(str(self.path))
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(str(self.path))
        if set(self.tagger.labels()) != {"0", "1", "2"}:
            raise ValueError("CRF lacks labels")
        np.savez(destination.with_suffix(".scaler.npz"), mean=self.scaler.mean, scale=self.scaler.scale)
        return self
    def predict(self, sequence):
        self.tagger.set(crf_items(self.scaler.transform(sequence.x), self.context))
        path = np.asarray(self.tagger.tag(), dtype=np.int8)
        p = np.array([[self.tagger.marginal(str(k), i) for k in range(3)] for i in range(len(path))])
        return normalize(p), path


CONFIGS = {
    "fuzzy_bare": ("fuzzy", {"radius": 2., "memory": 0.}),
    "fuzzy_r1_m02": ("fuzzy", {"radius": 1., "memory": .2}),
    "fuzzy_r2_m02": ("fuzzy", {"radius": 2., "memory": .2}),
    "fuzzy_r2_m05": ("fuzzy", {"radius": 2., "memory": .5}),
    "hmm_bare": ("hmm", {"weight": 1., "floor": .05}),
    "hmm_soft": ("hmm", {"weight": .25, "floor": .05}),
    "hmm_regularized": ("hmm", {"weight": .25, "floor": .2}),
    "crf_bare": ("crf", {"c2": 1., "context": 0}),
    "crf_context_c01": ("crf", {"c2": .1, "context": 5}),
    "crf_context_c1": ("crf", {"c2": 1., "context": 5}),
    "crf_context_c10": ("crf", {"c2": 10., "context": 5}),
}


def build(name):
    family, kwargs = CONFIGS[name]
    return {"fuzzy": Fuzzy, "hmm": GaussianHMM, "crf": LinearCRF}[family](**kwargs)


def normalize(p):
    p = np.maximum(np.asarray(p, dtype=float), 1e-9)
    return p/p.sum(axis=1, keepdims=True)


def temperature(p, value):
    return softmax(np.log(np.maximum(p, 1e-9))/value, axis=1)


def nll(p, y):
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())


def fit_temperature(p, y):
    # Temperature is a single parameter, fitted on held-out source-match labels only.
    result = minimize_scalar(lambda log_t: nll(temperature(p, np.exp(log_t)), y),
                             bounds=(np.log(.2), np.log(10)), method="bounded")
    value = float(np.exp(result.x))
    return value if nll(temperature(p, value), y) <= nll(p, y) else 1.0


def probability_metrics(p, y, prediction):
    valid = y < 3
    p, y, prediction = p[valid], y[valid], prediction[valid]
    conf = p[np.arange(len(y)), prediction]
    correct = prediction == y
    bins, ece = [], 0.
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (conf >= low) & ((conf < high) if high < 1 else (conf <= high))
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        avg = float(conf[mask].mean()) if count else None
        if count:
            ece += count/len(y)*abs(accuracy-avg)
        bins.append({"low": float(low), "high": float(high), "count": count,
                     "accuracy": accuracy, "confidence": avg})
    risk = []
    order = np.argsort(-conf, kind="stable")
    for coverage in (1., .9, .75, .5):
        selected = order[:max(1, int(len(order)*coverage))]
        risk.append({"coverage": coverage, "error_rate": float(1-correct[selected].mean())})
    return {"nll": nll(p, y), "brier": float(((p-np.eye(3)[y])**2).sum(axis=1).mean()),
            "ece": float(ece), "mean_confidence": float(conf.mean()),
            "frame_accuracy": float(correct.mean()), "bins": bins, "risk_coverage": risk}


def fit_weights(probabilities, y):
    best, best_loss = np.full(3, 1/3), float("inf")
    # Coarse, predeclared simplex; small penalty discourages fitting one noisy calibration tail.
    for i in range(11):
        for j in range(11-i):
            weights = np.array([i, j, 10-i-j])/10
            combined = sum(w*p for w, p in zip(weights, probabilities))
            loss = nll(combined, y)+.01*np.sum((weights-1/3)**2)
            if loss < best_loss:
                best, best_loss = weights, loss
    return best
