"""C3: local Transformer and factorized alive/team head, fixed 8s support."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import sha256
from .probabilistic import NAMES, STRIDE, Standardizer, temperature, transition_parameters
from .probabilistic_run import fusion
from .run import save_json
from .temporal import TCNConfig, calibrate, chunk_ranges, validate_decoder
from .team_temporal import TeamConfig, TeamTCN, TeamTemporalBundle, prepare


@dataclass(frozen=True)
class C3Config:
    backbone: str = 'transformer'
    head: str = 'had'
    epochs: int = 8
    chunk_samples: int = 1024
    learning_rate: float = .001
    weight_decay: float = .001
    seed: int = 7

    def __post_init__(self):
        if self.backbone not in ('tcn', 'transformer') or self.head not in ('had', 'factorized'):
            raise ValueError('Unknown C3 backbone/head')
        TCNConfig(22, (1,), self.epochs, self.chunk_samples, self.learning_rate, self.weight_decay, self.seed)


def joint_log_probabilities(logits):
    """Two logits: alive, home conditional on alive; order H/A/D."""
    alive, home = logits.unbind(-1)
    return torch.stack([F.logsigmoid(alive)+F.logsigmoid(home),
                        F.logsigmoid(alive)+F.logsigmoid(-home), F.logsigmoid(-alive)], -1)


class LocalBlock(nn.Module):
    """Pre-norm attention with relative offsets, no global attention matrix."""
    radius = 25
    width = 24
    heads = 4

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(self.width)
        self.qkv = nn.Linear(self.width, 3*self.width)
        self.project = nn.Linear(self.width, self.width)
        self.relative = nn.Parameter(torch.zeros(self.heads, 2*self.radius+1))
        self.norm2 = nn.LayerNorm(self.width)
        self.ff = nn.Sequential(nn.Linear(self.width, 64), nn.GELU(), nn.Linear(64, self.width))

    def forward(self, x):
        n, r = len(x), self.radius
        q, k, v = self.qkv(self.norm1(x)).view(n, 3, self.heads, self.width//self.heads).unbind(1)
        # [time, heads, channels, offsets]; padded keys are explicitly excluded.
        k = F.pad(k.permute(1, 2, 0), (r, r)).unfold(-1, 2*r+1, 1).permute(2, 0, 1, 3)
        v = F.pad(v.permute(1, 2, 0), (r, r)).unfold(-1, 2*r+1, 1).permute(2, 0, 1, 3)
        logits = torch.einsum('nhd,nhdw->nhw', q, k)/(self.width//self.heads)**.5
        positions = torch.arange(n, device=x.device)[:, None]+torch.arange(-r, r+1, device=x.device)[None]
        valid = (positions >= 0)&(positions < n)
        weights = (logits+self.relative).masked_fill(~valid[:, None], -torch.inf).softmax(-1)
        h = torch.einsum('nhw,nhdw->nhd', weights, v).reshape(n, self.width)
        x = x+self.project(h)
        return x+self.ff(self.norm2(x))


class LocalTransformer(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.project = nn.Linear(len(NAMES), 24)
        self.blocks = nn.Sequential(LocalBlock(), LocalBlock())
        self.output = nn.Linear(24, outputs)

    def forward(self, x):
        return self.output(self.blocks(self.project(x)))


class C3Model(nn.Module):
    radius = 50

    def __init__(self, config):
        super().__init__()
        self.head = config.head
        if config.backbone == 'tcn':
            self.backbone = TeamTCN(TeamConfig())
            if config.head == 'factorized':
                self.backbone.output = nn.Linear(22, 2)
        else:
            self.backbone = LocalTransformer(3 if config.head == 'had' else 2)

    def forward(self, x):
        return self.backbone(x)

    def probabilities(self, logits):
        return logits.softmax(-1) if self.head == 'had' else joint_log_probabilities(logits).exp()

    def loss(self, logits, truth):
        if self.head == 'had':
            return F.cross_entropy(logits, truth, reduction='sum')
        return F.nll_loss(joint_log_probabilities(logits), truth, reduction='sum')


class C3Bundle(TeamTemporalBundle):
    def raw(self, seq):
        x = self.scaler.transform(seq.x).astype(np.float32)
        if x.ndim != 2 or x.shape[1] != len(NAMES) or not np.isfinite(x).all() or not len(x):
            raise ValueError('Invalid C3 input')
        result = np.empty((len(x), 3), dtype=float)
        self.model.eval()
        with torch.inference_mode():
            for left, right, start, stop in chunk_ranges(len(x), self.config['model']['chunk_samples'], 50):
                logits = self.model(torch.from_numpy(x[left:right]).to(self.device))[start-left:stop-left]
                result[start:stop] = self.model.probabilities(logits).cpu().numpy()
        return result

    def stages(self, seq):
        raw = self.raw(seq)
        emission = temperature(raw, self.config['emission_temperature'])
        chain, labels = fusion([emission], [1.], np.asarray(self.config['transition']), np.asarray(self.config['prior']), .25)
        values = {'raw': raw, 'emission': emission, 'chain': chain,
                  'final': temperature(chain, self.config['chain_temperature'])}
        return ({k: np.repeat(p, STRIDE, axis=0)[:seq.frames] for k, p in values.items()},
                np.repeat(labels, STRIDE)[:seq.frames])

    def predict_prepared(self, seq):
        stages, labels = self.stages(seq)
        return stages['final'], labels

    def predict(self, data):
        return self.predict_prepared(prepare(data))

    @classmethod
    def load(cls, folder, device='cpu'):
        folder = Path(folder)
        hashes = json.loads((folder/'complete.json').read_text())['artifact_sha256']
        if set(hashes) != {'bundle.json', 'model.pt', 'scaler.npz'}:
            raise ValueError('Incomplete C3 artifact')
        for name, digest in hashes.items():
            if sha256(folder/name) != digest:
                raise ValueError('C3 artifact changed: '+name)
        config = json.loads((folder/'bundle.json').read_text())
        if (config['schema'] != 'team-temporal-c3-v1' or config['names'] != list(NAMES) or
                config['stride'] != STRIDE or config['classes'] != ['H', 'A', 'D']):
            raise ValueError('Incompatible C3 schema')
        validate_decoder(config)
        model = C3Model(C3Config(**config['model']))
        model.load_state_dict(torch.load(folder/'model.pt', map_location='cpu', weights_only=True))
        scaler = Standardizer()
        with np.load(folder/'scaler.npz', allow_pickle=False) as z:
            scaler.mean, scaler.scale = z['mean'].copy(), z['scale'].copy()
        if (scaler.mean.shape != (len(NAMES),) or scaler.scale.shape != (len(NAMES),) or
                not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.scale).all() or (scaler.scale <= 0).any()):
            raise ValueError('Invalid C3 scaler')
        return cls(model, scaler, config, device)


def fit(training, calibration, clean, sources, records, folder, config=C3Config(), device='cpu'):
    if any(s.y is None for s in training+calibration+clean) or set(np.concatenate([s.y for s in training])) != {0, 1, 2}:
        raise ValueError('Complete external training/calibration references required')
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    scaler = Standardizer().fit(training)
    xs = [scaler.transform(s.x).astype(np.float32) for s in training]
    model = C3Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    chunks = [(i, *b) for i, x in enumerate(xs) for b in chunk_ranges(len(x), config.chunk_samples, model.radius)]
    history = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        total, frames = 0., 0
        for index in rng.permutation(len(chunks)):
            i, left, right, start, stop = chunks[index]
            y = torch.from_numpy(training[i].y[start:stop].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(xs[i][left:right]).to(device))[start-left:stop-left]
            loss = model.loss(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite C3 training loss')
            (loss/config.chunk_samples).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach().cpu()); frames += len(y)
        history.append(total/frames)
        print(f'C3 {config.backbone}/{config.head} epoch {epoch+1}: {history[-1]:.5f}', flush=True)
    fit_seconds = time.perf_counter()-started
    transition, prior = transition_parameters(clean)
    cfg = {'schema': 'team-temporal-c3-v1', 'names': list(NAMES),
           'stride': STRIDE, 'classes': ['H', 'A', 'D'], 'model': asdict(config),
           'training_matches': list(sources), 'partitions': records,
           'transition': transition.tolist(), 'prior': prior.tolist(), 'chain_strength': .25,
           'parameters': sum(p.numel() for p in model.parameters()),
           'receptive_field_samples': 1+2*model.radius, 'span_seconds': 2*model.radius*STRIDE/25,
           'fit_seconds': fit_seconds, 'training_nll': history, 'device': device,
           'checkpoint_selection': 'Final fixed epoch; no held-out selection'}
    bundle = C3Bundle(model, scaler, cfg, device)
    started = time.perf_counter()
    cfg['emission_temperature'], cfg['chain_temperature'] = calibrate(bundle.raw, calibration, transition, prior)
    cfg['calibration_seconds'] = time.perf_counter()-started
    bundle.save(folder)
    return bundle
