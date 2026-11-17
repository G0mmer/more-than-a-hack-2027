"""Causal 2D kinematic evidence, not a physical simulator or possession labels.

Requires stable per-player tracks (the DFL adapter provides them). Never use
these derivatives on independently permuted per-frame detector slots.
"""
import warnings
import numpy as np

TEAM_NAMES=('speed_mean','collective_speed','motion_coherence','motion_coverage',
            'near_ball_relative_speed','near_ball_closing_speed','near_ball_velocity_cosine',
            'near_ball_projected_distance','near_ball_motion_valid')
NAMES=('ball_motion_valid','ball_speed_02s','ball_accel_valid','ball_accel_02s') + tuple(
    f'{team}_{name}' for team in ('H','A') for name in TEAM_NAMES)


def velocity(points, fps=25., lag=5, maximum_speed=60., check_steps=False):
    points=np.asarray(points,dtype=float)
    valid=np.isfinite(points).all(axis=-1)
    if check_steps:
        step_speed = np.linalg.norm(np.diff(points, axis=0), axis=-1) * fps
        valid = valid.copy()
        valid[1:] &= np.isfinite(step_speed) & (step_speed <= maximum_speed)
    bad=np.concatenate([np.zeros_like(valid[:1],dtype=np.int64),np.cumsum(~valid,axis=0)],axis=0)
    out=np.full_like(points,np.nan)
    if len(points)<=lag:return out
    uninterrupted=(bad[lag+1:]-bad[:-lag-1])==0
    delta=(points[lag:]-points[:-lag])*(fps/lag)
    usable=uninterrupted&(np.linalg.norm(delta,axis=-1)<=maximum_speed)
    out[lag:]=np.where(usable[...,None],delta,np.nan)
    return out


def features(data, check_steps=False):
    ball=np.asarray(data.ball,dtype=float); players=np.asarray(data.players,dtype=float)
    bv=velocity(ball,data.fps,maximum_speed=60., check_steps=check_steps)
    pv=velocity(players,data.fps,maximum_speed=15., check_steps=check_steps)
    ba=velocity(bv,data.fps,maximum_speed=300.)
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
        # Symmetric weighting: all nearby candidates, no arbitrary argmin tie.
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
