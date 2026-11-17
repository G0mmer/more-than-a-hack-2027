"""Matched 12.5 Hz development screen of motion and training noise."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path

import numpy as np

from .data import sha256
from .evaluation import official, evaluate_period, aggregate
from .kinematics import features as motion_features, NAMES as MOTION_NAMES
from .multimatch_run import MATCHES, partitions, load_completed
from .probabilistic import STRIDE, temperature, fit_temperature
from .probabilistic_run import fusion
from .robust import RobustCRF, mask_ball, sequence
from .robust_run import read_match
from .stress import perturb


def make_sequence(data, truth, motion):
    seq = sequence(data, False, truth)
    if motion:
        seq = replace(seq, x=np.column_stack((seq.x, motion_features(data)[::STRIDE])))
    return seq


def train_views(data, seed, noise):
    masked = mask_ball(data, .4, seed)[0]
    if not noise:
        return data, masked
    noisy = perturb(data, 'noise_0.3m', seed + 7200)[0]
    # Each instant receives one clean-coordinate and one noisy-coordinate view.
    blocks = (np.arange(len(data.ball)) // 250 + seed) % 2 == 0
    def blend(view, selected):
        ball = np.where(selected[:, None], noisy.ball, data.ball)
        ball = np.where(np.isfinite(view.ball), ball, np.nan)
        return replace(view, ball=ball, players=np.where(selected[:, None, None], noisy.players, data.players))
    return blend(data, blocks), blend(masked, ~blocks)


def run(data, baseline, out, held=MATCHES[0]):
    out.mkdir(parents=True, exist_ok=False)
    if held not in MATCHES: raise ValueError("Unknown development match")
    sources = [match for match in MATCHES if match != held]
    frozen = load_completed(baseline / ('leave_out_'+held) / 'augmentation')
    cfg = frozen.config
    assert cfg['training_matches'] == sources and cfg['stride'] == 2
    protocol = {'held':held,'fit_matches':sources,'design':'One development fold; paired 2x2 motion and noise ablation',
        'train':'Original 80/20 source splits with purge, clean + masked40; for noise variants alternate 10s blocks, complementary views, 0.3m Gaussian noise',
        'calibration':'Original clean and masked source tails, same decoder and temperature procedure',
        'eval':['clean','missing40','noise_0.3m','noise_plus_missing40'],'eval_seeds':[1000,8100],
        'baseline_hash':sha256(baseline/('leave_out_'+held)/'augmentation'/'model.crfsuite'),
        'code_sha256':sha256(__file__)}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    datasets={}
    for match in MATCHES:
        datasets[match]=read_match(data,match)[0]
        print('Loaded',match,flush=True)
    train,cal,records=partitions(datasets,sources)
    assert records == cfg['partitions']
    trans,prior=np.array(cfg['transition']),np.array(cfg['prior'])
    halves=datasets[held]
    cases={'clean':halves}
    cases['missing40']=[(mask_ball(d,.4,1000+i)[0],y) for i,(d,y) in enumerate(halves)]
    cases['noise_0.3m']=[(perturb(d,'noise_0.3m',8100+i)[0],y) for i,(d,y) in enumerate(halves)]
    cases['noise_plus_missing40']=[(mask_ball(d,.4,1000+i)[0],y) for i,(d,y) in enumerate(cases['noise_0.3m'])]
    pm=official()
    rows=[]
    for name,motion,noise in [('baseline',False,False),('noise',False,True),('motion',True,False),('motion_noise',True,True)]:
        if name=='baseline':
            model=frozen.model
            emission=cfg['emission_temperature']
            chain=cfg['chain_temperature']
        else:
            print('Training',name,flush=True)
            seqs=[[],[]]
            for d,y,seed in train:
                for index,view in enumerate(train_views(d,seed,noise)):
                    seqs[index].append(make_sequence(view,y,motion))
            model=RobustCRF(False)
            if motion: model.names=tuple(model.names)+tuple(MOTION_NAMES)
            model.fit(seqs[0]+seqs[1],out/name)
            del seqs
            cal_seqs=[make_sequence(d,y,motion) for d,y,_ in cal]+[make_sequence(mask_ball(d,.4,seed)[0],y,motion) for d,y,seed in cal]
            raw=[model.predict(seq)[0] for seq in cal_seqs]
            truth=np.concatenate([seq.y for seq in cal_seqs])
            emission=fit_temperature(np.concatenate(raw),truth)
            chained=[fusion([temperature(p,emission)],[1.],trans,prior,.25)[0] for p in raw]
            chain=fit_temperature(np.concatenate(chained),truth)
            (out/name/'config.json').write_text(json.dumps({'names':model.names,'emission_temperature':emission,'chain_temperature':chain,'transition':trans.tolist(),'prior':prior.tolist(),'strength':.25,'stride':2},indent=2))
        for scenario,examples in cases.items():
            scores=[]
            for d,y in examples:
                seq=make_sequence(d,None,motion)
                p=model.predict(seq)[0]
                path=fusion([temperature(p,emission)],[1.],trans,prior,.25)[1]
                scores.append(evaluate_period(pm,y,np.repeat(path,STRIDE)[:len(y)]))
            row={'variant':name,'scenario':scenario,**aggregate(pm,scores)}
            rows.append(row)
            (out/'scores.json').write_text(json.dumps(rows,indent=2))
            print(name,scenario,round(row['spell_f1'],4),round(row['f1_D'],4),flush=True)
        del model
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('data','baseline','out'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument("--held",choices=MATCHES,default=MATCHES[0])
    run(**vars(parser.parse_args()))

