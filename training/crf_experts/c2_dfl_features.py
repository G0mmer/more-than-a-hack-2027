"""C2 feature adapter: only the two position derivatives differ between A/B.

DFL Tracking columns must be stable tracks, with missing rows retained.
Call separately per half/crop. All derivatives are computed at 25 Hz.
"""
from dataclasses import replace
import numpy as np
from possession_lab import kinematics
from possession_lab.probabilistic import NAMES as GEOMETRY_NAMES, STRIDE
from possession_lab.robust import sequence
from c2_velocity import local_linear_velocity

NAMES = tuple(GEOMETRY_NAMES) + tuple(kinematics.NAMES)


def derivatives(data, arm):
    if data.fps != 25 or np.any(np.diff(data.frame_ids) != 1):
        raise ValueError("Expected a contiguous 25 Hz timeline with stable track columns")
    ball = np.asarray(data.ball, dtype=float)
    players = np.asarray(data.players, dtype=float)
    if arm == "A":
        bv = kinematics.velocity(ball, data.fps, maximum_speed=60.)
        pv = kinematics.velocity(players, data.fps, maximum_speed=15.)
    elif arm == "B":
        bx, by, _ = local_linear_velocity(ball, fps=data.fps, maximum_speed=60.)
        px, py, _ = local_linear_velocity(players, fps=data.fps, maximum_speed=15.)
        bv = np.stack((bx, by), axis=-1)
        pv = np.stack((px, py), axis=-1)
    else:
        raise ValueError("arm must be A or B")
    # Deliberately unchanged for BOTH arms; no global velocity replacement.
    ba = kinematics.velocity(bv, data.fps, lag=5, maximum_speed=300.)
    return bv, pv, ba


def pool_motion(data, bv, pv, ba):
    """Same aggregation/arithmetic/order as possession_lab.kinematics.features."""
    ball=np.asarray(data.ball,dtype=float); players=np.asarray(data.players,dtype=float)
    blocks=[np.column_stack([np.isfinite(bv).all(1),np.linalg.norm(bv,axis=1)/60,
                            np.isfinite(ba).all(1),np.linalg.norm(ba,axis=1)/300])]
    for team in (0,1):
        p=players[:,data.teams==team]; v=pv[:,data.teams==team]
        valid=np.isfinite(v).all(2); count=valid.sum(1)
        speed=np.linalg.norm(v,axis=2)
        vsum=np.nansum(v,axis=1)
        avg=np.divide(vsum,count[:,None],out=np.zeros_like(vsum),where=count[:,None]>0)
        mean_speed=np.divide(np.nansum(speed,axis=1),count,out=np.zeros(len(ball)),where=count>0)
        collective=np.linalg.norm(avg,axis=1)
        coherence=np.divide(collective,mean_speed,out=np.zeros(len(ball)),where=mean_speed>1e-8)
        displacement=ball[:,None]-p
        distance=np.linalg.norm(displacement,axis=2)
        near=np.isfinite(distance)&(distance<=3.)&valid&np.isfinite(bv).all(1)[:,None]
        weights=np.where(near,np.exp(-np.nan_to_num(distance,nan=1e6)/2),0.)
        denom=weights.sum(1)
        relative=bv[:,None]-v
        rspeed=np.linalg.norm(relative,axis=2)
        closing=-np.sum(displacement*relative,axis=2)/np.maximum(distance,.1)
        cosine=np.sum(bv[:,None]*v,axis=2)/np.maximum(np.linalg.norm(bv,axis=1)[:,None]*speed,.01)
        projected=np.linalg.norm(displacement+.5*relative,axis=2)
        def weighted(values):
            return np.divide((np.nan_to_num(values)*weights).sum(1),denom,
                             out=np.zeros(len(ball)),where=denom>0)
        blocks.append(np.column_stack([mean_speed/15,collective/15,coherence,
            count/max(1,p.shape[1]),weighted(rspeed)/75,weighted(closing)/75,
            weighted(cosine),weighted(projected)/40,denom>0]))
    return np.nan_to_num(np.column_stack(blocks),nan=0,posinf=0,neginf=0).astype(np.float32)


def availability_counts(data, bv, pv, ba):
    observed_ball = np.isfinite(data.ball).all(-1)
    observed_players = np.isfinite(data.players).all(-1)
    masks = {
        "ball_velocity": (np.isfinite(bv).all(-1), observed_ball),
        "player_velocity": (np.isfinite(pv).all(-1), observed_players),
        "ball_acceleration": (np.isfinite(ba).all(-1), observed_ball),
    }
    for team, label in ((0, "home_velocity"), (1, "away_velocity")):
        columns = data.teams == team
        masks[label] = (np.isfinite(pv[:, columns]).all(-1), observed_players[:, columns])
    return {
        grid: {name: {"valid": int(valid[::step].sum()),
                      "observed": int(observed[::step].sum()),
                      "total": int(valid[::step].size)}
               for name, (valid, observed) in masks.items()}
        for grid, step in (("25Hz", 1), ("12.5Hz", STRIDE))
    }


def motion_features(data, arm, diagnostics=False):
    bv, pv, ba = derivatives(data, arm)
    x = pool_motion(data, bv, pv, ba)
    return (x, availability_counts(data, bv, pv, ba)) if diagnostics else x


def make_sequence(data, truth, arm, diagnostics=False):
    base = sequence(data, False, truth)
    motion = motion_features(data, arm, diagnostics)
    x, counts = motion if diagnostics else (motion, None)
    seq = replace(base, x=np.column_stack((base.x, x[::STRIDE])))
    if seq.x.shape[1] != 35 or len(NAMES) != 35:
        raise ValueError("C1 feature schema changed: expected 13 + 22 = 35")
    return (seq, counts) if diagnostics else seq
