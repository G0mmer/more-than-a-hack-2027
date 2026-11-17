"""Generate a candidate CSV from a verified external-model bundle."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .be_ha_gate import visibility_gate
from .data import Tracking, sha256
from .ensemble import decode
from .ensemble_predict import EnsembleBundle
from .final_round import read_json
from .moe import prepare
from .restart_supervision import ARMS, Expert, infer
from .stoppage_evidence import auxiliary, evidence
from .submission import read_atd, write_validated
from .symmetry import SymmetryAdapter
from .team_identity import Specialist, decode_p1_team, lock_dead, player_features
from .team_shape import STRIDE
from .team_temporal import TeamTemporalBundle

TEAM_MODES = ('anchor', 'p1', 'hybrid')


def configure_cpu():
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)


def compose(data, anchor, p1, specialist, restart, variants):
    config = anchor.tcn.config
    emissions, quality = prepare(data, anchor.tcn, anchor.a, anchor.d)
    posterior, _ = anchor.gate.predict(emissions, quality)
    _, base = decode(posterior, config)
    length = len(data.ball)
    base_full = np.repeat(base,2)[:length]
    dead_paths = {'moe_d':base_full}
    if any(dead!='moe_d' for dead,team in variants):
        features = {k:v.astype(np.float32) for k,v in evidence(data).items()} | auxiliary(data)
        dead_paths, _ = infer(features,base_full,restart)
    live_paths = {'anchor':base_full}
    masks = {}
    if any(team!='anchor' for dead,team in variants):
        if p1 is None:
            raise ValueError('P1 weights are required for team composition')
        p1_path, p1_mask = decode_p1_team(data,posterior,base,config,p1)
        live_paths['p1'] = np.repeat(p1_path,2)[:length]
        masks['p1'] = int(p1_mask.sum())
    if any(team=='hybrid' for dead,team in variants):
        if specialist is None:
            raise ValueError('Movement specialist weights are required for hybrid composition')
        motion_mask, _, _ = visibility_gate(data)
        motion_mask = motion_mask & (base<2)
        if motion_mask.any():
            _, motion = specialist.predict(player_features(data))
            home = np.interp(np.arange(0,length,2),np.arange(0,length,STRIDE),motion)
            hybrid = lock_dead(posterior,p1_path,home,motion_mask,config,.50)
        else:
            hybrid = p1_path
        live_paths['hybrid'] = np.repeat(hybrid,2)[:length]
        masks['motion'] = int(motion_mask.sum())
    result = {}
    for dead,team in variants:
        is_dead = dead_paths[dead]==2
        labels = np.where(is_dead,2,live_paths[team])
        if not np.array_equal(labels==2,is_dead):
            raise ValueError('Team expert changed the selected D path')
        result[dead,team] = labels
    return result,masks


class CandidateBundle:
    def __init__(self, path):
        path = Path(path).resolve()
        receipt = read_json(path/'manifest.json')
        if receipt['schema']!='submission-candidates-v1':
            raise ValueError('Unexpected candidate bundle')
        files = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}
        if files-{'manifest.json'}!=set(receipt['sha256']):
            raise ValueError('Bundle file list differs from the manifest')
        for name,digest in receipt['sha256'].items():
            file = (path/name).resolve()
            if not file.is_relative_to(path) or sha256(file)!=digest:
                raise ValueError('Changed model artifact: '+name)
        self.anchor = EnsembleBundle(path/'anchor','moe_dead')
        if receipt['anchor_signature']!=self.anchor.model_signature:
            raise ValueError('Unexpected anchor signature')
        self.p1 = SymmetryAdapter(TeamTemporalBundle.load(path/'p1'))
        self.specialist = Specialist(path/'team_motion.pt')
        self.restart = {k:Expert(v) for k,v in read_json(path/'restart.json').items()}
        self.manifest = receipt

    def predict(self,data,variants):
        return compose(data,self.anchor,self.p1,self.specialist,self.restart,variants)


def predict_match(bundle,data_dir,variants):
    from games import GAME, GAMES, half_video_blocks, require
    game = GAMES[GAME]
    meta,raw = require(['atd_meta','atd_raw'],data_dir=data_dir)
    ball,players,teams = read_atd(meta,raw,game['n_video_frames'],game['pitch'])
    output = {variant:np.full(len(ball),'X') for variant in variants}
    covered = np.zeros(len(ball),bool)
    for half,lo,hi in half_video_blocks(GAME):
        if not 0<=lo<=hi<len(ball) or covered[lo:hi+1].any():
            raise ValueError('Invalid half frame axis')
        if not np.isfinite(players[lo:hi+1]).all(axis=2).any():
            raise ValueError(f'No player observations in half {half}')
        ids = np.arange(lo,hi+1)
        tracking = Tracking('firstHalf' if half==1 else 'secondHalf',ids,
                            ball[ids],players[ids],teams,game['pitch'])
        predicted,_ = bundle.predict(tracking,variants)
        if set(predicted) != set(variants):
            raise ValueError('Missing or unexpected prediction variant')
        covered[ids] = True
        for variant,labels in predicted.items():
            if labels.shape!=(len(ids),) or not np.isin(labels,(0,1,2)).all():
                raise ValueError('Invalid half prediction')
            output[variant][ids] = np.array(list('HAD'))[labels]
    return output


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','data-dir','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--dead',choices=ARMS,required=True)
    parser.add_argument('--team',choices=TEAM_MODES,required=True)
    a = parser.parse_args()
    if a.out.exists() or a.out.suffix.lower()!='.csv':
        parser.error('Use a new CSV output path')
    configure_cpu()
    variant = (a.dead,a.team)
    path = predict_match(CandidateBundle(a.bundle),a.data_dir,[variant])[variant]
    write_validated(a.out,path)
    print('Validated',len(path),'rows, SHA-256',sha256(a.out),flush=True)
