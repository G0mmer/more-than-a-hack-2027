"""New portable all-development-match refit; historical C3/C4 artifacts stay frozen.

Reuses the original C2 training/calibration arithmetic without invoking its owner
launchers or changing their fixed four-source checks. Produces a new manifest.
"""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch

from .baseline_suite import initialize
from .data import sha256
from .ensemble import MotionBundle,MOTION_FEATURES
from .moe import fit as fit_gate
from .moe_run import training_rows
from .multimatch_run import MATCHES,partitions
from .probabilistic import fit_temperature,temperature
from .probabilistic_run import fusion
from .robust import RobustCRF
from .robust_run import read_match
from .run import save_json
from .team_temporal import TeamTemporalBundle


def artifact_hashes(folder):
    return {n:sha256(folder/n) for n in ('config.json','model.crfsuite','scaler.npz')}


def run(args):
    repo=Path(__file__).resolve().parents[1]
    helpers=repo/'training/crf_experts'
    initialize(args.out,'new-five-match-motion-refit',dict(matches=list(MATCHES),
        training='Original C2 complementary 250-frame motion_noise views; source80/cal20; purge125',
        helper_sha256={p.name:sha256(p) for p in helpers.glob('*.py')},
        historical_models_modified=False,selection_not_based_on_refit_metrics=True))
    # These imports expose pure math helpers; their guarded owner CLIs are not run.
    sys.path.insert(0,str(helpers))
    recipe=importlib.import_module('run_c2_dfl')
    dead=importlib.import_module('c2_dead_specialist')
    legacy=importlib.import_module('c3_dfl')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    datasets,files={},[]
    for match in MATCHES:
        print('READ EXTERNAL '+match,flush=True)
        datasets[match],_,sources=read_match(args.dfl_root,match);files.extend(sources)
    save_json(args.out/'inputs.json',files)
    before={m:[legacy.observation_record(d,y) for d,y in halves] for m,halves in datasets.items()}
    train,cal,records=partitions(datasets,list(MATCHES))
    transition,prior,_=legacy.clean_transitions(train)
    common=dict(schema='motion-five-match-refit-v1',names=list(MOTION_FEATURES),
        training_matches=list(MATCHES),partitions=records,
        fold={'held':None,'sources':list(MATCHES),'refit':'all-development-matches-v1'},
        history=False,stride=2,input_hz=25,context_samples=[-5,0,5],
        velocity_backend='five_frame',acceleration='five_frame_velocity_difference')
    print('FIT FINAL MOTION A',flush=True)
    started=time.perf_counter()
    sequences=recipe.training_sequences(train,'A')
    a=RobustCRF(False);a.names=MOTION_FEATURES
    a.fit(sequences,args.out/'motion_A');del sequences
    cal_sequences=recipe.calibration_sequences(cal,'A')
    raw=[a.predict(s)[0] for s in cal_sequences]
    y=np.concatenate([s.y for s in cal_sequences])
    ta=fit_temperature(np.concatenate(raw),y)
    chained=[fusion([temperature(p,ta)],[1.],transition,prior,.25)[0] for p in raw]
    tc=fit_temperature(np.concatenate(chained),y)
    config=dict(common,variant='motion_noise',emission_temperature=float(ta),chain_temperature=float(tc),
        transition=transition.tolist(),prior=prior.tolist(),strength=.25,
        native_schema=legacy.check_native_schema(a.tagger,('0','1','2')))
    save_json(args.out/'motion_A/config.json',config)
    ah=artifact_hashes(args.out/'motion_A')
    save_json(args.out/'motion_A/complete.json',dict(artifact_sha256=ah,role='new_final_A'))
    del raw,chained,cal_sequences
    print('FIT FINAL D SPECIALIST',flush=True)
    sequences=dead.training_sequences(train)
    binary=dead.BinaryCRF().fit(sequences,args.out/'dead_specialist',a.scaler);del sequences
    calibration=dead.calibration_sequences(cal)
    raw=[binary.predict_proba(s) for s in calibration]
    td=fit_temperature(np.concatenate(raw),np.concatenate([s.y for s in calibration]))
    dc=dict(common,model_role='binary_dead_specialist',labels=['0','1'],label_meanings={'0':'alive','1':'D'},
        reference_A_artifact_sha256=ah,binary_emission_temperature=float(td),
        native_schema=legacy.check_native_schema(binary.tagger,('0','1')))
    save_json(args.out/'dead_specialist/config.json',dc)
    save_json(args.out/'dead_specialist/complete.json',dict(artifact_sha256=artifact_hashes(args.out/'dead_specialist'),
        reference_A_artifact_sha256=ah,role='new_final_D'))
    del raw,calibration
    after={m:[legacy.observation_record(d,y) for d,y in halves] for m,halves in datasets.items()}
    if before!=after:raise ValueError('Refit changed source observations')
    shutil.copytree(args.tcn,args.out/'tcn')
    tcn=TeamTemporalBundle.load(args.out/'tcn')
    ma,md=MotionBundle(args.out/'motion_A'),MotionBundle(args.out/'dead_specialist',True)
    print('FIT FINAL GATES',flush=True)
    e,x,y=training_rows(cal,tcn,ma,md)
    np.savez_compressed(args.out/'gate_training_inputs.npz',experts=e,quality=x,truth=y)
    for kind in ('quality','dead'):
        gate,history=fit_gate(e,x,y,kind)
        gate.save(args.out/('gate_'+kind),dict(training_matches=list(MATCHES),held_out=None,
            partitions=records,n_training_rows=len(y),final_source_loss=history[-1],
            expert_artifact_sha256={'tcn':json.loads((args.out/'tcn/complete.json').read_text())['artifact_sha256'],
                'motion_A':ma.hashes,'motion_A_D':md.hashes}))
    marker=dict(schema='all-five-expert-package-v1',seconds=time.perf_counter()-started,
        training_matches=list(MATCHES),selected_from_external_cv_only=True,
        artifact_sha256={str(p.relative_to(args.out)).replace('\\','/'):sha256(p)
            for p in args.out.rglob('*') if p.is_file() and 'source_snapshot' not in p.parts})
    save_json(args.out/'complete.json',marker)
    print('FINAL PACKAGE COMPLETE',args.out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dfl-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tcn',type=Path,required=True,help='Anchor TCN folder whose emissions train the gates')
    run(p.parse_args())


if __name__=='__main__':main()
