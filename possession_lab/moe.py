"""Source-calibration-trained constant ensemble and quality-dependent softmax gate.

Experts and their temperatures remain frozen. Weights mix aligned native H/A/D
emissions at 12.5 Hz, followed by ONE shared decoder outside this module.
"""
from pathlib import Path
import json
import numpy as np
import torch

from .data import sha256
from .ensemble import emission_candidates, probabilities, compose_dead
from .kinematics import features as motion_features
from .probabilistic import geometry
from .run import save_json

EXPERTS=('tcn','motion_A','motion_A_D')
FEATURES=('ball_seen','player_fraction','nearby_count','nearby_speed','ball_speed_1f',
          'ball_velocity_valid','ball_speed_5f','ball_acceleration_valid','ball_acceleration',
          'confidence_tcn','confidence_A','confidence_AD','entropy_tcn','entropy_A','entropy_AD',
          'D_tcn','D_A','D_AD','disagreement_tcn_A','disagreement_tcn_AD')


def prepare(data,tcn,a,d):
    emissions=emission_candidates(tcn,a,d,data)
    experts=np.stack([emissions[name] for name in EXPERTS],axis=1)
    return experts,quality_features(experts,geometry(data)[::2],motion_features(data)[::2])


def quality_features(experts,base,motion):
    """The original gate inputs, also usable with verified cached emissions."""
    confidence=experts.max(2)
    entropy=-(experts*np.log(np.maximum(experts,1e-12))).sum(2)/np.log(3)
    x=np.column_stack((base[:,7],np.clip(base[:,10],0,1.5),base[:,8],base[:,9],base[:,3],
        motion[:,:4],confidence,entropy,experts[:,:,2],
        np.abs(experts[:,0]-experts[:,1]).sum(1)/2,np.abs(experts[:,0]-experts[:,2]).sum(1)/2))
    if x.shape!=(len(experts),len(FEATURES)) or not np.isfinite(x).all():
        raise ValueError('Invalid MoE quality features')
    return x


def validate(experts,x):
    experts=np.asarray(experts,dtype=float);x=np.asarray(x,dtype=float)
    if experts.ndim!=3 or experts.shape[1:]!=(len(EXPERTS),3):raise ValueError('Wrong expert axes')
    for i in range(len(EXPERTS)):probabilities(experts[:,i])
    if x.shape!=(len(experts),len(FEATURES)) or not np.isfinite(x).all():raise ValueError('Wrong quality features')
    return experts,x


class Gate:
    def __init__(self,weights,bias,mean,scale,kind):
        self.weights,self.bias,self.mean,self.scale=map(np.asarray,(weights,bias,mean,scale))
        self.kind=kind
        if (kind not in ('constant','quality','dead','dead_constant') or self.weights.shape!=(len(FEATURES),3) or
                self.bias.shape!=(3,) or self.mean.shape!=(len(FEATURES),) or self.scale.shape!=self.mean.shape or
                any(not np.isfinite(v).all() for v in (self.weights,self.bias,self.mean,self.scale)) or (self.scale<=0).any()):
            raise ValueError('Invalid gate artifact')

    def predict(self,experts,x,available=None):
        experts,x=validate(experts,x)
        z=np.clip((x-self.mean)/self.scale,-10,10)
        logits=z@self.weights+self.bias
        if available is None:available=np.ones(logits.shape,dtype=bool)
        available=np.asarray(available,dtype=bool)
        if available.shape==(3,):available=np.broadcast_to(available,logits.shape)
        if available.shape!=logits.shape or not available.any(1).all():raise ValueError('No available expert')
        logits=np.where(available,logits,-np.inf)
        weights=np.exp(logits-logits.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True)
        p=(experts*weights[:,:,None]).sum(1)
        if self.kind.startswith('dead'):
            # Separate ball-alive decision from the identity of the possessing team.
            # Prefer TCN H/A odds; on an unavailable TCN use the first live expert.
            reference=experts[np.arange(len(experts)),available.argmax(1)]
            composed=compose_dead(reference,p[:,2])
            p=np.where((available.sum(1)==1)[:,None],p,composed)
        return probabilities(p),weights

    def save(self,folder,metadata):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
        np.savez(folder/'gate.npz',weights=self.weights,bias=self.bias,mean=self.mean,scale=self.scale)
        save_json(folder/'config.json',dict(schema='native-emission-moe-v1',kind=self.kind,
            experts=list(EXPERTS),features=list(FEATURES),**metadata))
        save_json(folder/'complete.json',{'artifact_sha256':{n:sha256(folder/n) for n in ('gate.npz','config.json')}})

    @classmethod
    def load(cls,folder):
        folder=Path(folder)
        hashes=json.loads((folder/'complete.json').read_text())['artifact_sha256']
        if set(hashes)!={'gate.npz','config.json'}:raise ValueError('Incomplete gate')
        for n,h in hashes.items():
            if sha256(folder/n)!=h:raise ValueError('Changed gate artifact')
        config=json.loads((folder/'config.json').read_text())
        if config['schema']!='native-emission-moe-v1' or config['experts']!=list(EXPERTS) or config['features']!=list(FEATURES):
            raise ValueError('Gate schema mismatch')
        with np.load(folder/'gate.npz',allow_pickle=False) as z:
            result=cls(z['weights'],z['bias'],z['mean'],z['scale'],config['kind'])
        return result,config


def fit(experts,x,truth,kind):
    experts,x=validate(experts,x)
    truth=np.asarray(truth)
    if truth.shape!=(len(x),) or set(np.unique(truth))!={0,1,2}:raise ValueError('Complete source H/A/D required')
    mean,scale=x.mean(0),np.maximum(x.std(0),.05)
    z=torch.from_numpy(np.clip((x-mean)/scale,-10,10).astype(np.float32))
    e=torch.from_numpy(experts.astype(np.float32))
    y=torch.from_numpy(truth.astype(np.int64))
    torch.manual_seed(7)
    learned=kind in ('quality','dead')
    w=torch.zeros((len(FEATURES),3),requires_grad=learned)
    b=torch.zeros(3,requires_grad=True)
    optimizer=torch.optim.AdamW([w,b] if learned else [b],lr=.03,weight_decay=.01)
    selected=e[torch.arange(len(e)),:,y]
    history=[]
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        alpha=torch.softmax(z@w+b,dim=1)
        if kind.startswith('dead'):
            q=(alpha*e[:,:,2]).sum(1).clamp(1e-7,1-1e-7)
            loss=torch.nn.functional.binary_cross_entropy(q,(y==2).to(q.dtype))
        else:
            loss=-torch.log((alpha*selected).sum(1).clamp_min(1e-9)).mean()
        # Small explicit shrinkage discourages unnecessary source-specific gates.
        objective=loss+1e-3*(w.square().mean()+b.square().mean())
        objective.backward();optimizer.step()
        history.append(float(loss.detach()))
    return Gate(w.detach().numpy(),b.detach().numpy(),mean,scale,kind),history
