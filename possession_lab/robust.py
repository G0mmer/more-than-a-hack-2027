"""Frozen CRF ablation: missing-ball augmentation and causal observation history."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pycrfsuite
from .data import Tracking
from .probabilistic import (NAMES, STRIDE, geometry, Standardizer, Sequence, normalize,
                            temperature, expand)
from .probabilistic_run import fusion
from .run import save_json

HISTORY = ("missing_age_5s", "seen_fraction_2s", "last_ball_x", "last_ball_y", "ever_seen")
VARIANTS = {"control": (False, False), "history": (True, False),
            "augmentation": (False, True), "augmentation_history": (True, True)}

def crop(data, start, stop):
    return replace(data, frame_ids=data.frame_ids[start:stop].copy(), ball=data.ball[start:stop].copy(),
                   players=data.players[start:stop].copy())

def mask_ball(data, fraction, seed, long=False):
    if not 0 <= fraction < 1:
        raise ValueError("Invalid missing fraction")
    rng = np.random.default_rng(seed)
    n = len(data.ball)
    mask = np.zeros(n, dtype=bool)
    target = int(n*fraction)
    while mask.sum() < target:
        length = int(rng.integers(50,201) if long else rng.integers(13,51))
        start = int(rng.integers(n))
        mask[start:min(n,start+length)] = True
    ball = data.ball.copy()
    ball[mask] = np.nan
    return replace(data,ball=ball), float(mask.mean())

def robust_geometry(data, history):
    x = geometry(data)
    if not history:
        return x
    valid = np.isfinite(data.ball).all(axis=1)
    n = len(valid)
    latest = np.maximum.accumulate(np.where(valid,np.arange(n),-1))
    known = latest >= 0
    age = np.where(known,np.minimum((np.arange(n)-latest)/25,5),5)/5
    held = np.zeros((n,2))
    held[known] = data.ball[latest[known]]/np.asarray(data.pitch)
    cs = np.r_[0,np.cumsum(valid)]
    first = np.maximum(0,np.arange(n)-49)
    coverage = (cs[np.arange(n)+1]-cs[first])/(np.arange(n)+1-first)
    return np.column_stack([x,age,coverage,held,known.astype(float)])

def sequence(data, history, truth=None):
    x = robust_geometry(data, history)
    # Dummy labels are structural only. No target labels are accepted by prediction.
    y = np.zeros(len(x),dtype=np.int8) if truth is None else np.asarray(truth)
    if len(y) != len(x) or not np.isin(y,[0,1,2]).all():
        raise ValueError("Training requires complete public reference")
    return Sequence(data.period,x[::STRIDE],y[::STRIDE],y,0,len(y))

def items(x, names):
    n = len(x)
    values = np.column_stack([np.ones(n)]+[x[np.clip(np.arange(n)+o,0,n-1)] for o in (-5,0,5)])
    keys = ["bias"]+[f"{o}:{name}" for o in (-5,0,5) for name in names]
    return pycrfsuite.ItemSequence([dict(zip(keys,map(float,row))) for row in values])

class RobustCRF:
    def __init__(self, history):
        self.history = bool(history)
        self.names = tuple(NAMES)+(HISTORY if history else ())
    def fit(self, seqs, folder):
        folder.mkdir(parents=True)
        self.scaler = Standardizer().fit(seqs)
        trainer = pycrfsuite.Trainer(verbose=False)
        for s in seqs:
            trainer.append(items(self.scaler.transform(s.x),self.names),[str(int(y)) for y in s.y])
        trainer.set_params({"c1":0,"c2":.1,"max_iterations":80,
                            "feature.possible_transitions":True,"feature.possible_states":True})
        trainer.train(str(folder/"model.crfsuite"))
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(str(folder/"model.crfsuite"))
        if set(self.tagger.labels()) != {"0","1","2"}:
            raise ValueError("CRF training lacks H/A/D")
        np.savez(folder/"scaler.npz",mean=self.scaler.mean,scale=self.scaler.scale)
        return self
    @classmethod
    def load(cls, folder, config):
        model = cls(config["history"])
        model.scaler = Standardizer()
        with np.load(folder/"scaler.npz",allow_pickle=False) as z:
            model.scaler.mean, model.scaler.scale = z["mean"].copy(),z["scale"].copy()
        if len(model.scaler.mean)!=len(model.names):
            raise ValueError("Feature shape mismatch")
        model.tagger = pycrfsuite.Tagger()
        model.tagger.open(str(folder/"model.crfsuite"))
        return model
    def predict(self, s):
        self.tagger.set(items(self.scaler.transform(s.x),self.names))
        y = np.asarray(self.tagger.tag(),dtype=np.int8)
        p = np.array([[self.tagger.marginal(str(k),i) for k in range(3)] for i in range(len(y))])
        return normalize(p),y

class Bundle:
    def __init__(self, model, config):
        self.model,self.config = model,config
    @classmethod
    def load(cls, folder):
        config = json.loads((folder/"bundle.json").read_text(encoding="utf-8"))
        if config["stride"] != STRIDE:
            raise ValueError("Stride mismatch")
        return cls(RobustCRF.load(folder,config),config)
    def predict(self, data):
        s = sequence(data,self.config["history"])
        p,_ = self.model.predict(s)
        p = temperature(p,self.config["emission_temperature"])
        q,y = fusion([p],[1.],np.asarray(self.config["transition"]),np.asarray(self.config["prior"]),.25)
        return expand(temperature(q,self.config["chain_temperature"]),s),expand(y,s)
