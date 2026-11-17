"""Small offline TCN on the frozen CRF feature contract (C1).

Every call sees one half or one source partition. Chunk halos preserve the
convolution's receptive field; neither training nor inference joins halves.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .data import sha256
from .probabilistic import (NAMES, STRIDE, Standardizer, expand,
                            fit_temperature, temperature, transition_parameters)
from .probabilistic_run import fusion
from .robust import mask_ball, sequence
from .run import save_json


@dataclass(frozen=True)
class TCNConfig:
    hidden: int = 32
    dilations: tuple = (1, 2, 4, 8)
    epochs: int = 8
    chunk_samples: int = 1024
    learning_rate: float = .001
    weight_decay: float = .001
    seed: int = 7

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.hidden, self.epochs, self.chunk_samples, self.seed)) or
                self.hidden < 1 or self.epochs < 1 or self.chunk_samples < 1 or self.seed < 0 or
                not self.dilations or any(type(d) is not int or d < 1 for d in self.dilations) or
                not np.isfinite([self.learning_rate, self.weight_decay]).all() or
                self.learning_rate <= 0 or self.weight_decay < 0):
            raise ValueError('Invalid TCN configuration')


def validate_decoder(config):
    transition, prior = np.asarray(config['transition']), np.asarray(config['prior'])
    temps = np.asarray([config['emission_temperature'], config['chain_temperature']], dtype=float)
    if (transition.shape != (3, 3) or prior.shape != (3,) or
            not np.isfinite(transition).all() or not np.isfinite(prior).all() or
            (transition <= 0).any() or (prior <= 0).any() or
            not np.allclose(transition.sum(1), 1) or not np.isclose(prior.sum(), 1) or
            not np.isfinite(temps).all() or (temps <= 0).any() or config['chain_strength'] != .25):
        raise ValueError('Invalid temporal decoder configuration')


class PossessionTCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.radius = sum(config.dilations)
        self.project = nn.Conv1d(len(NAMES), config.hidden, 1)
        self.temporal = nn.Sequential(*[
            layer for dilation in config.dilations for layer in (
                nn.Conv1d(config.hidden, config.hidden, 3,
                          dilation=dilation, padding=dilation), nn.ReLU())])
        self.output = nn.Conv1d(config.hidden, 3, 1)

    def forward(self, x):
        """[batch, time, the same 13 CRF features] -> H/A/D logits."""
        x = torch.relu(self.project(x.transpose(1, 2)))
        return self.output(self.temporal(x)).transpose(1, 2)


def chunk_ranges(length, core, radius):
    if length < 1 or core < 1 or radius < 0:
        raise ValueError('Invalid sequence/chunk length')
    for start in range(0, length, core):
        stop = min(length, start + core)
        yield max(0, start-radius), min(length, stop+radius), start, stop


def raw_probabilities(model, x, chunk_samples=1024, device='cpu'):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(NAMES) or not np.isfinite(x).all():
        raise ValueError('Expected finite CRF geometry features')
    probabilities = np.empty((len(x), 3), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for left, right, start, stop in chunk_ranges(len(x), chunk_samples, model.radius):
            logits = model(torch.from_numpy(x[left:right].copy())[None].to(device))
            p = logits[0, start-left:stop-left].softmax(-1)
            probabilities[start:stop] = p.cpu().numpy()
    return probabilities


def calibrate(raw, calibration, transition, prior):
    probabilities = [raw(s) for s in calibration]
    truth = np.concatenate([s.y for s in calibration])
    emission = fit_temperature(np.concatenate(probabilities), truth)
    chained = [fusion([temperature(p, emission)], [1.], transition, prior, .25)[0]
               for p in probabilities]
    chain = fit_temperature(np.concatenate(chained), truth)
    return emission, chain


class TemporalBundle:
    """Tracking-only inference; labels cannot be supplied to predict()."""
    def __init__(self, model, scaler, config, device='cpu'):
        self.model = model.to(device).eval()
        self.scaler, self.config, self.device = scaler, config, device

    def raw(self, seq):
        return raw_probabilities(self.model, self.scaler.transform(seq.x),
                                 self.config['tcn']['chunk_samples'], self.device)

    def predict(self, data):
        seq = sequence(data, False)
        p = temperature(self.raw(seq), self.config['emission_temperature'])
        q, y = fusion([p], [1.], np.asarray(self.config['transition']),
                      np.asarray(self.config['prior']), .25)
        return expand(temperature(q, self.config['chain_temperature']), seq), expand(y, seq)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        save_json(folder/'bundle.json', self.config)
        # Tensors and primitives only; load explicitly uses weights_only=True.
        torch.save({k: v.detach().cpu() for k, v in self.model.state_dict().items()}, folder/'model.pt')
        np.savez(folder/'scaler.npz', mean=self.scaler.mean, scale=self.scaler.scale)
        save_json(folder/'complete.json', {'artifact_sha256': {
            name: sha256(folder/name) for name in ('bundle.json', 'model.pt', 'scaler.npz')}})

    @classmethod
    def load(cls, folder, device='cpu'):
        folder = Path(folder)
        hashes = json.loads((folder/'complete.json').read_text(encoding='utf-8'))['artifact_sha256']
        if set(hashes) != {'bundle.json', 'model.pt', 'scaler.npz'}:
            raise ValueError('Incomplete TCN artifact')
        for name, digest in hashes.items():
            if sha256(folder/name) != digest:
                raise ValueError('TCN artifact changed: '+name)
        config = json.loads((folder/'bundle.json').read_text(encoding='utf-8'))
        if (config['schema'] != 'team-temporal-c1-v1' or config['stride'] != STRIDE or
                config['names'] != list(NAMES) or config['classes'] != ['H', 'A', 'D']):
            raise ValueError('Incompatible TCN feature/class schema')
        validate_decoder(config)
        settings = dict(config['tcn'])
        settings['dilations'] = tuple(settings['dilations'])
        model = PossessionTCN(TCNConfig(**settings))
        model.load_state_dict(torch.load(folder/'model.pt', map_location='cpu', weights_only=True))
        scaler = Standardizer()
        with np.load(folder/'scaler.npz', allow_pickle=False) as z:
            scaler.mean, scaler.scale = z['mean'].copy(), z['scale'].copy()
        if (scaler.mean.shape != (len(NAMES),) or scaler.scale.shape != (len(NAMES),) or
                not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.scale).all() or
                (scaler.scale <= 0).any()):
            raise ValueError('Invalid TCN scaler')
        return cls(model, scaler, config, device)


def fit_tcn(train, cal, records, sources, folder, config=TCNConfig(), device='cpu'):
    """Same clean + missing40 views, standardizer and calibration as CRF augmentation."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    clean = [sequence(d, False, y) for d, y, _ in train]
    training = clean + [sequence(mask_ball(d, .4, seed)[0], False, y) for d, y, seed in train]
    calibration = ([sequence(d, False, y) for d, y, _ in cal] +
                   [sequence(mask_ball(d, .4, seed)[0], False, y) for d, y, seed in cal])
    if set(np.concatenate([s.y for s in training])) != {0, 1, 2}:
        raise ValueError('Training requires all three external H/A/D states')
    scaler = Standardizer().fit(training)
    xs = [scaler.transform(s.x).astype(np.float32) for s in training]
    model = PossessionTCN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    chunks = [(i, *bounds) for i, x in enumerate(xs)
              for bounds in chunk_ranges(len(x), config.chunk_samples, model.radius)]
    history = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        total_loss, total_frames = 0., 0
        for index in rng.permutation(len(chunks)):
            i, left, right, start, stop = chunks[index]
            x = torch.from_numpy(xs[i][left:right].copy())[None].to(device)
            y = torch.from_numpy(training[i].y[start:stop].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)[0, start-left:stop-left]
            loss_sum = nn.functional.cross_entropy(logits, y, reduction='sum')
            if not torch.isfinite(loss_sum):
                raise RuntimeError('Nonfinite TCN loss')
            # A short tail contributes in proportion to its frame count.
            (loss_sum/config.chunk_samples).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float(loss_sum.detach().cpu())
            total_frames += len(y)
        history.append(total_loss/total_frames)
        print(f'TCN epoch {epoch+1}/{config.epochs}: NLL={history[-1]:.5f}', flush=True)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    fit_seconds = time.perf_counter()-started
    transition, prior = transition_parameters(clean)
    cfg = {'schema': 'team-temporal-c1-v1', 'names': list(NAMES), 'classes': ['H', 'A', 'D'],
           'stride': STRIDE, 'history': False, 'augmentation': True, 'tcn': asdict(config),
           'receptive_field_samples': 1+2*model.radius,
           'receptive_field_span_seconds': 2*model.radius*STRIDE/25,
           'training_matches': list(sources), 'partitions': records,
           'transition': transition.tolist(), 'prior': prior.tolist(), 'chain_strength': .25,
           'parameters': sum(p.numel() for p in model.parameters()), 'device': device,
           'fit_seconds': fit_seconds, 'training_nll': history,
           'checkpoint_selection': 'Final fixed epoch; no held-out match or calibration-based selection'}
    bundle = TemporalBundle(model, scaler, cfg, device)
    started = time.perf_counter()
    cfg['emission_temperature'], cfg['chain_temperature'] = calibrate(
        bundle.raw, calibration, transition, prior)
    cfg['calibration_seconds'] = time.perf_counter()-started
    bundle.save(folder)
    return bundle
