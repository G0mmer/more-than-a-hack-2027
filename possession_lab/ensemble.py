"""Explicit emission-stage adapters for the temporal, motion and binary D models.

Historical checkpoints remain unchanged. All combinations are made at half-local
12.5 Hz BEFORE one shared decoder; final TCN marginals are never averaged with
native CRF marginals. No reference labels are accepted by this module.
"""
import json
from pathlib import Path

import numpy as np
import pycrfsuite

from .data import sha256
from .kinematics import NAMES as MOTION_NAMES, features as motion_features
from .probabilistic import NAMES, geometry, temperature
from .probabilistic_run import fusion
from .robust import items
from .team_temporal import prepare

MOTION_FEATURES = (*NAMES, *MOTION_NAMES)
CANDIDATES = ('tcn', 'motion_A', 'motion_A_D', 'mean_emission',
              'visibility_gate', 'tcn_D', 'mean_emission_D')


def probabilities(p, columns=3):
    p = np.asarray(p, dtype=float)
    if (p.ndim != 2 or not len(p) or p.shape[1] != columns or
            not np.isfinite(p).all() or (p < 0).any() or
            not np.allclose(p.sum(1), 1, atol=1e-7, rtol=0)):
        raise ValueError('Invalid normalized probability rows')
    return p


def compose_dead(p, q):
    p = probabilities(p)
    q = np.asarray(q)
    if q.shape != (len(p),) or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError('Invalid D probabilities')
    alive = p[:, 0]+p[:, 1]
    home = np.divide(p[:, 0], alive, out=np.full(len(p), .5), where=alive > 0)
    return np.column_stack(((1-q)*home, (1-q)*(1-home), q))


def decode(p, config):
    return fusion([probabilities(p)], [1.], np.asarray(config['transition']),
                  np.asarray(config['prior']), .25)


class MotionBundle:
    """Load a C3 motion A or binary D directory, with its original checksums."""
    def __init__(self, folder, binary=False):
        self.folder, self.binary = Path(folder), binary
        marker = json.loads((self.folder/'complete.json').read_text(encoding='utf-8'))
        hashes = marker['artifact_sha256']
        if set(hashes) != {'config.json', 'model.crfsuite', 'scaler.npz'}:
            raise ValueError('Incomplete motion model')
        for name, digest in hashes.items():
            if sha256(self.folder/name) != digest:
                raise ValueError('Motion artifact changed: '+name)
        self.config = json.loads((self.folder/'config.json').read_text(encoding='utf-8'))
        if self.config['names'] != list(MOTION_FEATURES):
            raise ValueError('Wrong motion feature names or order')
        self.hashes, self.marker = hashes, marker
        with np.load(self.folder/'scaler.npz', allow_pickle=False) as z:
            self.mean, self.scale = z['mean'].copy(), z['scale'].copy()
        if (self.mean.shape != (35,) or self.scale.shape != (35,) or
                not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or
                (self.scale <= 0).any()):
            raise ValueError('Invalid motion scaler')
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(str(self.folder/'model.crfsuite'))
        labels = ['0', '1'] if binary else ['0', '1', '2']
        if sorted(self.tagger.labels()) != labels:
            raise ValueError('Unexpected native label meanings')
        schema = self.config['native_schema']
        if sorted(self.tagger.info().attributes) != schema['native_attribute_names']:
            raise ValueError('Native feature schema changed')
        self.temp = self.config['binary_emission_temperature' if binary else 'emission_temperature']
        if not np.isfinite(self.temp) or self.temp <= 0:
            raise ValueError('Invalid motion temperature')

    def raw_features(self, data):
        return np.column_stack((geometry(data)[::2], motion_features(data)[::2]))

    def emissions(self, data, features=None):
        x = self.raw_features(data) if features is None else features
        if x.shape != ((len(data.ball)+1)//2, 35) or not np.isfinite(x).all():
            raise ValueError('Invalid half-local motion features')
        self.tagger.set(items(np.clip((x-self.mean)/self.scale, -10, 10), MOTION_FEATURES))
        columns = 2 if self.binary else 3
        raw = np.array([[self.tagger.marginal(str(k), i) for k in range(columns)] for i in range(len(x))])
        return probabilities(temperature(raw/raw.sum(1, keepdims=True), self.temp), columns)


def validate_pair(motion, specialist):
    if specialist is None:
        return
    if (not specialist.binary or motion.binary or
            specialist.marker.get('reference_A_artifact_sha256') != motion.hashes or
            specialist.config['fold'] != motion.config['fold'] or
            not np.array_equal(specialist.mean, motion.mean) or
            not np.array_equal(specialist.scale, motion.scale)):
        raise ValueError('D specialist is not bound to this A model')


def emission_candidates(tcn, motion, specialist, data):
    """Return native emissions without reading labels or decoding twice."""
    spec = tcn.config.get('model', {})
    seq = prepare(data, spec.get('representation', 'geometry') != 'geometry',
                  feature_set=spec.get('feature_set', 'geometry'))
    pt = probabilities(temperature(tcn.raw(seq), tcn.config['emission_temperature']))
    if motion is None:
        return {'tcn': pt}
    validate_pair(motion, specialist)
    x = motion.raw_features(data)
    pa = motion.emissions(data, x)
    if pa.shape != pt.shape:
        raise ValueError('Model timeline mismatch')
    mean = (pt+pa)/2
    visible = np.isfinite(data.ball[::2]).all(1)
    result = {'tcn': pt, 'motion_A': pa, 'mean_emission': mean,
              'visibility_gate': np.where(visible[:, None], mean, pt)}
    if specialist is not None:
        q = specialist.emissions(data, x)[:, 1]
        result.update(motion_A_D=compose_dead(pa, q), tcn_D=compose_dead(pt, q),
                      mean_emission_D=compose_dead(mean, q))
    return result


def predict_candidates(tcn, motion, specialist, data, names=None):
    emissions = emission_candidates(tcn, motion, specialist, data)
    predictions = {}
    for name, p in emissions.items():
        if names is not None and name not in names:
            continue
        config = motion.config if name in ('motion_A', 'motion_A_D') else tcn.config
        q, y = decode(p, config)
        predictions[name] = (np.repeat(q, 2, axis=0)[:len(data.ball)],
                             np.repeat(y, 2)[:len(data.ball)])
    return predictions
