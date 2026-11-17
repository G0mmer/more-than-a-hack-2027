"""Label-free inference for a pinned all-five expert package.

Accepts either one external Tracking NPZ or a tactical data bundle. Never reads
reference labels. The original possession_lab.submission baseline is unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from .data import sha256
from .ensemble import MotionBundle,emission_candidates,decode
from .moe import Gate,prepare
from .team_temporal import TeamTemporalBundle
from .temporal_predict import load_tracking
from .submission import read_atd,predict_halves,write_validated


class EnsembleBundle:
    def __init__(self,folder,variant):
        if variant not in ('tcn_D','moe_quality','moe_dead'):raise ValueError('Unknown ensemble variant')
        folder=Path(folder).resolve()
        marker=json.loads((folder/'complete.json').read_text())
        if marker['schema']!='all-five-expert-package-v1':raise ValueError('Wrong expert package')
        for name,digest in marker['artifact_sha256'].items():
            path=folder/name
            if not path.resolve().is_relative_to(folder) or sha256(path)!=digest:raise ValueError('Changed package artifact')
        self.tcn=TeamTemporalBundle.load(folder/'tcn')
        self.a,self.d=MotionBundle(folder/'motion_A'),MotionBundle(folder/'dead_specialist',True)
        self.variant=variant
        self.gate=None
        if variant in ('moe_quality','moe_dead'):
            self.gate,config=Gate.load(folder/('gate_'+variant.removeprefix('moe_')))
            sources=self.tcn.config['training_matches']
            if config['training_matches']!=sources or self.a.config['training_matches']!=sources:
                raise ValueError('Expert/gate source mismatch')
        signature=dict(variant=variant,
            tcn=json.loads((folder/'tcn/complete.json').read_text())['artifact_sha256'],
            motion_A=self.a.hashes,dead_specialist=self.d.hashes,
            gate=sha256(folder/('gate_'+variant.removeprefix('moe_'))/'complete.json') if self.gate else None)
        self.model_signature=hashlib.sha256(json.dumps(signature,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def predict(self,data):
        if self.gate is not None:
            e,x=prepare(data,self.tcn,self.a,self.d);p,_=self.gate.predict(e,x)
        else:
            p=emission_candidates(self.tcn,self.a,self.d,data)['tcn_D']
        q,y=decode(p,self.tcn.config)
        return np.repeat(q,2,axis=0)[:len(data.ball)],np.repeat(y,2)[:len(data.ball)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--variant',choices=('tcn_D','moe_quality','moe_dead'),required=True)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--tracking',type=Path);inputs.add_argument('--data-dir',type=Path)
    parser.add_argument('--match',help='Explicit match identifier for a Tracking NPZ')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Use a new output path')
    if args.tracking and not args.match:parser.error('--match is required with --tracking')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model=EnsembleBundle(args.bundle,args.variant)
    if args.tracking:
        if args.out.suffix.lower()!='.npz':parser.error('Tracking NPZ inference writes .npz')
        data=load_tracking(args.tracking);p,y=model.predict(data)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.out,schema='expert-ensemble-predictions-v1',frame=data.frame_ids,
            match=args.match,half=data.period,source='tracking',model_variant=args.variant,
            model_signature=model.model_signature,
            p_H=p[:,0],p_A=p[:,1],p_D=p[:,2],label=np.array(['H','A','D'])[y],
            ball_available=np.isfinite(data.ball).all(1),
            player_fraction=np.isfinite(data.players).all(2).sum(1)/max(1,len(data.teams)),
            fps=25.,model_rate_hz=12.5,probability_stage='post-chain, before optional marginal calibration')
    else:
        if args.out.suffix.lower()!='.csv':parser.error('Tactical bundle inference writes .csv')
        from games import GAME,GAMES,require,half_video_blocks
        game=GAMES[GAME]
        meta,raw=require(['atd_meta','atd_raw'],data_dir=args.data_dir)
        ball,players,teams=read_atd(meta,raw,game['n_video_frames'],game['pitch'])
        labels=predict_halves(model,ball,players,teams,game['pitch'],half_video_blocks(GAME))
        write_validated(args.out,labels)
    print('VALIDATED',args.out,'SHA-256',sha256(args.out),flush=True)


if __name__=='__main__':main()
