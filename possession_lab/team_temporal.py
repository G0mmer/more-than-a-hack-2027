"""C2: fixed-capacity temporal context and permutation-invariant player sets."""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .data import sha256
from .kinematics import features as motion_features, NAMES as MOTION_NAMES
from .probabilistic import NAMES, STRIDE, Standardizer, geometry, temperature, transition_parameters
from .probabilistic_run import fusion
from .run import save_json
from .temporal import TCNConfig, calibrate, chunk_ranges, validate_decoder


# Eight layers in every variant; dense receptive support without dilation holes.
# Nominal 2s is quantized to a symmetric 25-sample window (1.92s span).
DILATIONS = {2: (1, 1, 1, 1, 1, 1, 3, 3),
             8: (1, 1, 2, 3, 5, 8, 13, 17),
             32: (1, 2, 4, 8, 16, 32, 64, 73)}
NODE_NAMES = ('pitch_centre_dx_20m', 'pitch_centre_dy_20m',
              'ball_dx_20m', 'ball_dy_20m', 'ball_distance_20m', 'ball_available',
              'nearest_opponent_20m', 'team_centre_dx_20m', 'team_centre_dy_20m')
REPRESENTATIONS = ('geometry', 'none', 'mean', 'deepsets', 'graph')


@dataclass(frozen=True)
class TeamConfig:
    context_seconds: int = 8
    representation: str = 'geometry'
    hidden: int = 22
    epochs: int = 8
    chunk_samples: int = 1024
    learning_rate: float = .001
    weight_decay: float = .001
    seed: int = 7
    feature_set: str = "geometry"

    def __post_init__(self):
        if self.feature_set not in ('geometry', 'motion', 'motion_masked'):
            raise ValueError('Unknown feature set')
        if self.feature_set != 'geometry' and self.representation != 'geometry':
            raise ValueError('Motion experiment uses the geometry representation')
        if self.context_seconds not in DILATIONS or self.representation not in REPRESENTATIONS:
            raise ValueError('Unknown context or player representation')
        TCNConfig(self.hidden, DILATIONS[self.context_seconds], self.epochs, self.chunk_samples,
                  self.learning_rate, self.weight_decay, self.seed)


@dataclass
class TeamInput:
    x: np.ndarray
    nodes: np.ndarray | None
    visible: np.ndarray | None
    teams: np.ndarray
    y: np.ndarray | None
    frames: int


def feature_names(feature_set):
    return NAMES if feature_set == 'geometry' else (*NAMES, *MOTION_NAMES)


def prepare(data, player_sets=False, truth=None, feature_set='geometry'):
    """One half/partition; no slot derivatives in the player-set experiment.

    The context experiment keeps all original CRF features. All four player-set
    variants replace nearby_speed (column 9) with zero, because its slot-wise
    derivative is not invariant to independently reordered detections.
    """
    safe = data
    for team in (0, 1):
        if not np.any(safe.teams == team):
            safe = replace(safe, teams=np.r_[safe.teams, team],
                           players=np.concatenate([safe.players, np.full((len(safe.ball), 1, 2), np.nan)], axis=1))
    if feature_set not in ('geometry', 'motion', 'motion_masked'):
        raise ValueError('Unknown feature set')
    masked = feature_set == 'motion_masked'
    x = geometry(safe, mask_motion=masked)[::STRIDE]
    if feature_set != 'geometry':
        x = np.column_stack((x, motion_features(safe, check_steps=masked)[::STRIDE]))
    y = None if truth is None else np.asarray(truth)
    if y is not None and (y.shape != (len(data.ball),) or not np.isin(y, [0, 1, 2]).all()):
        raise ValueError('Aligned complete external H/A/D reference required')
    if y is not None:
        y = y[::STRIDE].copy()
    if not player_sets:
        return TeamInput(x, None, None, data.teams.copy(), y, len(data.ball))
    x[:, 9] = 0.
    players = np.asarray(data.players[::STRIDE], dtype=float)
    ball = np.asarray(data.ball[::STRIDE], dtype=float)
    visible = np.isfinite(players).all(-1)
    ball_seen = np.isfinite(ball).all(-1)
    nodes = np.zeros((*visible.shape, len(NODE_NAMES)), dtype=np.float32)
    # Bounded temporary pair-distance arrays, including on full match halves.
    for start in range(0, len(x), 2048):
        stop = min(len(x), start+2048)
        valid = visible[start:stop]
        p = np.where(valid[..., None], players[start:stop], 0.)
        b = np.where(ball_seen[start:stop, None], ball[start:stop], 0.)
        relative = b[:, None]-p
        relative = np.where(ball_seen[start:stop, None, None], relative, 0.)
        block = nodes[start:stop]
        block[..., :2] = (p-np.asarray(data.pitch)/2)/20
        block[..., 2:4] = np.clip(relative/20, -6, 6)
        block[..., 4] = np.minimum(np.linalg.norm(relative, axis=-1)/20, 6)
        block[..., 5] = ball_seen[start:stop, None]
        distances = np.linalg.norm(p[:, :, None]-p[:, None, :], axis=-1)
        for team in (0, 1):
            own, other = data.teams == team, data.teams != team
            count = valid[:, own].sum(1)
            centre = (p[:, own]*valid[:, own, None]).sum(1)/np.maximum(count[:, None], 1)
            block[:, own, 7:9] = (p[:, own]-centre[:, None])/20
            if other.any():
                d = np.where(valid[:, None, other], distances[:, own][:, :, other], np.inf)
                nearest = d.min(-1)
            else:
                nearest = np.full((stop-start, int(own.sum())), np.inf)
            block[:, own, 6] = np.minimum(nearest/20, 3)
        block[~valid] = 0.
    return TeamInput(x, nodes, visible, data.teams.copy(), y, len(data.ball))


class PlayerSetEncoder(nn.Module):
    """Same learned parameter budget for mean, DeepSets and a relation graph."""
    def __init__(self, mode):
        super().__init__()
        if mode not in ('mean', 'deepsets', 'graph'):
            raise ValueError('Unknown pooling mode')
        self.mode = mode
        self.phi = nn.Sequential(nn.Linear(len(NODE_NAMES), 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(24, 8), nn.ReLU())

    def forward(self, nodes, visible, teams):
        # Frames are independent sets; no player IDs or slot trajectories.
        masks = [visible & (teams == team)[None] for team in (0, 1)]
        counts = torch.stack([m.sum(1) for m in masks], dim=1)

        def pool(values):
            return torch.stack([(values*m[..., None]).sum(1)/m.sum(1).clamp_min(1)[:, None]
                                for m in masks], dim=1)

        if self.mode == 'mean':
            h = self.phi(pool(nodes))*(counts > 0)[..., None]
        else:
            node_h = self.phi(nodes)*visible[..., None]
            if self.mode == 'graph':
                # Separate teammate/opponent messages; own self-edge is retained.
                distances = torch.linalg.vector_norm(nodes[:, :, None, :2]-nodes[:, None, :, :2], dim=-1)*20
                weights = torch.exp(-distances/10)*(distances <= 20)
                weights = weights*visible[:, :, None]*visible[:, None, :]
                same = teams[:, None] == teams[None, :]
                messages = []
                for relation in (same, ~same):
                    edges = weights*relation[None]
                    edges = edges/edges.sum(-1, keepdim=True).clamp_min(1e-8)
                    messages.append(torch.bmm(edges, node_h))
                node_h = self.rho(torch.cat([node_h, *messages], dim=-1))*visible[..., None]
                h = pool(node_h)
            else:
                h = pool(node_h)
        if self.mode != 'graph':
            h = self.rho(torch.cat([h, h, h.flip(1)], dim=-1))
        h = h*(counts > 0)[..., None]
        return torch.cat([h.flatten(1), counts.to(h.dtype)/11], dim=-1)


class TeamTCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.radius = sum(DILATIONS[config.context_seconds])
        self.players = (PlayerSetEncoder(config.representation)
                        if config.representation in ('mean', 'deepsets', 'graph') else None)
        self.project = nn.Linear(len(feature_names(config.feature_set))+(18 if self.players else 0), config.hidden)
        self.temporal = nn.ModuleList([
            nn.Conv1d(config.hidden, config.hidden, 3, padding=d, dilation=d)
            for d in DILATIONS[config.context_seconds]])
        self.output = nn.Linear(config.hidden, 3)

    def forward(self, x, nodes=None, visible=None, teams=None):
        if self.players is not None:
            x = torch.cat([x, self.players(nodes, visible, teams)], dim=-1)
        h = torch.relu(self.project(x))
        h = h.T[None]
        for layer in self.temporal:
            h = torch.relu(h+layer(h))
        return self.output(h[0].T)


def forward_chunk(model, seq, scaled, left, right, device):
    x = torch.from_numpy(scaled[left:right]).to(device)
    if model.players is None:
        return model(x)
    return model(x, torch.from_numpy(seq.nodes[left:right]).to(device),
                 torch.from_numpy(seq.visible[left:right]).to(device),
                 torch.from_numpy(seq.teams).to(device))


class TeamTemporalBundle:
    def __init__(self, model, scaler, config, device='cpu'):
        self.model, self.scaler, self.config, self.device = model.to(device).eval(), scaler, config, device

    def raw(self, seq):
        x = self.scaler.transform(seq.x).astype(np.float32)
        if x.ndim != 2 or x.shape[1] != len(feature_names(self.config['model'].get('feature_set', 'geometry'))) or not np.isfinite(x).all():
            raise ValueError('Invalid team-temporal features')
        output = np.empty((len(x), 3), dtype=float)
        self.model.eval()
        with torch.inference_mode():
            for left, right, start, stop in chunk_ranges(len(x), self.config['model']['chunk_samples'], self.model.radius):
                logits = forward_chunk(self.model, seq, x, left, right, self.device)
                output[start:stop] = logits[start-left:stop-left].softmax(-1).cpu().numpy()
        return output

    def predict_prepared(self, seq):
        p = temperature(self.raw(seq), self.config['emission_temperature'])
        q, y = fusion([p], [1.], np.asarray(self.config['transition']), np.asarray(self.config['prior']), .25)
        return (np.repeat(temperature(q, self.config['chain_temperature']), STRIDE, axis=0)[:seq.frames],
                np.repeat(y, STRIDE)[:seq.frames])

    def predict(self, data):
        return self.predict_prepared(prepare(data, self.config['model']['representation'] != 'geometry', feature_set=self.config['model'].get('feature_set', 'geometry')))

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        save_json(folder/'bundle.json', self.config)
        torch.save({k: v.detach().cpu() for k, v in self.model.state_dict().items()}, folder/'model.pt')
        np.savez(folder/'scaler.npz', mean=self.scaler.mean, scale=self.scaler.scale)
        save_json(folder/'complete.json', {'artifact_sha256': {
            name: sha256(folder/name) for name in ('bundle.json', 'model.pt', 'scaler.npz')}})

    @classmethod
    def load(cls, folder, device='cpu'):
        folder = Path(folder)
        hashes = json.loads((folder/'complete.json').read_text())['artifact_sha256']
        if set(hashes) != {'bundle.json', 'model.pt', 'scaler.npz'}:
            raise ValueError('Incomplete C2 artifact')
        for name, digest in hashes.items():
            if sha256(folder/name) != digest:
                raise ValueError('C2 artifact changed: '+name)
        config = json.loads((folder/'bundle.json').read_text())
        names = feature_names(config['model'].get('feature_set', 'geometry'))
        if (config['schema'] != 'team-temporal-c2-v1' or config['names'] != list(names) or
                config['node_names'] != list(NODE_NAMES) or config['stride'] != STRIDE or
                config['classes'] != ['H', 'A', 'D']):
            raise ValueError('Incompatible C2 schema')
        validate_decoder(config)
        model = TeamTCN(TeamConfig(**config['model']))
        model.load_state_dict(torch.load(folder/'model.pt', map_location='cpu', weights_only=True))
        scaler = Standardizer()
        with np.load(folder/'scaler.npz', allow_pickle=False) as z:
            scaler.mean, scaler.scale = z['mean'].copy(), z['scale'].copy()
        if (scaler.mean.shape != (len(names),) or scaler.scale.shape != (len(names),) or
                not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.scale).all() or (scaler.scale <= 0).any()):
            raise ValueError('Invalid C2 scaler')
        return cls(model, scaler, config, device)


def fit(training, calibration, clean, sources, records, folder, config=TeamConfig(), device='cpu', *, epoch_training=None):
    if any(s.y is None for s in training+calibration+clean) or set(np.concatenate([s.y for s in training])) != {0, 1, 2}:
        raise ValueError('Complete external training/calibration references required')
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    scaler = Standardizer().fit(training)
    xs = [scaler.transform(s.x).astype(np.float32) for s in training]
    model = TeamTCN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    chunks = [(i, *b) for i, x in enumerate(xs) for b in chunk_ranges(len(x), config.chunk_samples, model.radius)]
    history = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        current = training if epoch_training is None else epoch_training(epoch)
        if len(current) != len(training) or any(
                s.x.shape != original.x.shape or s.frames != original.frames or
                s.y is None or s.y.shape != original.y.shape or
                not np.isin(s.y, [0, 1, 2]).all()
                for s, original in zip(current, training)):
            raise ValueError('Epoch augmentation must preserve sequence shapes and valid labels')
        epoch_xs = xs if epoch_training is None else [
            scaler.transform(s.x).astype(np.float32) for s in current]
        model.train()
        total, frames = 0., 0
        for index in rng.permutation(len(chunks)):
            i, left, right, start, stop = chunks[index]
            y = torch.from_numpy(current[i].y[start:stop].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_chunk(model, current[i], epoch_xs[i], left, right, device)[start-left:stop-left]
            loss = nn.functional.cross_entropy(logits, y, reduction='sum')
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite C2 training loss')
            (loss/config.chunk_samples).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach().cpu()); frames += len(y)
        history.append(total/frames)
        print(f'C2 {config.representation}/{config.context_seconds}s epoch {epoch+1}: {history[-1]:.5f}', flush=True)
    fit_seconds = time.perf_counter()-started
    transition, prior = transition_parameters(clean)
    cfg = {'schema': 'team-temporal-c2-v1', 'names': list(feature_names(config.feature_set)), 'node_names': list(NODE_NAMES),
           'stride': STRIDE, 'classes': ['H', 'A', 'D'], 'model': asdict(config),
           'training_matches': list(sources), 'partitions': records,
           'transition': transition.tolist(), 'prior': prior.tolist(), 'chain_strength': .25,
           'parameters': sum(p.numel() for p in model.parameters()),
           'receptive_field_samples': 1+2*model.radius, 'span_seconds': 2*model.radius*STRIDE/25,
           'fit_seconds': fit_seconds, 'training_nll': history, 'device': device,
           'checkpoint_selection': 'Final fixed epoch; no held-out selection'}
    bundle = TeamTemporalBundle(model, scaler, cfg, device)
    started = time.perf_counter()
    cfg['emission_temperature'], cfg['chain_temperature'] = calibrate(bundle.raw, calibration, transition, prior)
    cfg['calibration_seconds'] = time.perf_counter()-started
    bundle.save(folder)
    return bundle
