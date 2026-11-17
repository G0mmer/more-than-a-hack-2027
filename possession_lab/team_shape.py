"""Ball-free, slot-invariant team geometry with a calibrated linear baseline."""
from dataclasses import replace
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from .probabilistic import fit_temperature, temperature

STRIDE=5
BASE_NAMES=('visible_fraction','centre_x_abs','centre_y_abs','std_x','std_y',
            'width_x_80','width_y_80','cov_xy_abs','opponent_dist_mean',
            'opponent_dist_p10','opponent_dist_p90','opponent_within_2m','opponent_within_5m')
TEAM_DIM=len(BASE_NAMES)*3

def shape_features(data):
    """5 Hz features: current set geometry and causal 1s/5s differences.

    Reflection-invariant in both pitch axes; no player IDs, velocities or ball.
    Team A/B swap is a permutation of the two equal-sized feature blocks.
    """
    players=np.asarray(data.players[::STRIDE],dtype=float)
    visible=np.isfinite(players).all(2)
    players=np.where(visible[:,:,None],players,np.nan)
    normalized=players/np.asarray(data.pitch)
    count=len(players); blocks=[]
    for team in (0,1):
        own=normalized[:,data.teams==team]
        own_m=players[:,data.teams==team]
        opposing=players[:,data.teams==1-team]
        if own.shape[1]==0:
            current=np.zeros((count,len(BASE_NAMES)))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                centre=np.nanmean(own,axis=1)
                sd=np.nanstd(own,axis=1)
                q=np.nanquantile(own,[.1,.9],axis=1)
                spread=q[1]-q[0]
                cov=np.abs(np.nanmean((own[:,:,0]-centre[:,0,None])*(own[:,:,1]-centre[:,1,None]),axis=1))
                if opposing.shape[1]:
                    dist=np.linalg.norm(own_m[:,:,None,:]-opposing[:,None,:,:],axis=3)
                    nearest=np.nanmin(dist,axis=2)
                else:
                    nearest=np.full(own.shape[:2],np.nan)
                valid=np.isfinite(nearest)
                close=[np.divide(((nearest<r)&valid).sum(1),valid.sum(1),
                          out=np.zeros(count),where=valid.sum(1)>0) for r in (2,5)]
                proximity=np.column_stack([np.nanmean(nearest,axis=1),
                               np.nanquantile(nearest,.1,axis=1),np.nanquantile(nearest,.9,axis=1)])/10
                current=np.column_stack([np.isfinite(own).all(2).sum(1)/11,
                         np.abs(centre-.5),sd,spread,cov,proximity,*close])
            current=np.nan_to_num(current,nan=0,posinf=0,neginf=0)
        index=np.arange(count)
        blocks.append(np.column_stack([current,current-current[np.maximum(0,index-5)],
                                       current-current[np.maximum(0,index-25)]]))
    return np.column_stack(blocks).astype(np.float32)

def swap_features(x):
    return np.concatenate([x[:,TEAM_DIM:],x[:,:TEAM_DIM]],axis=1)

def swap_labels(y):
    return np.choose(y,[1,0,2])

class ShapeModel:
    def fit(self,x,y,cal_x,cal_y):
        if not np.isin(y,[0,1,2]).all() or not np.isin(cal_y,[0,1,2]).all():
            raise ValueError('Complete external H/A/D references required')
        # Balanced team identity, not balanced class frequencies.
        x=np.concatenate([x,swap_features(x)]); y=np.concatenate([y,swap_labels(y)])
        self.mean=x.mean(0); self.scale=x.std(0); self.scale[self.scale<1e-5]=1
        z=np.column_stack([np.ones(len(x)),(x-self.mean)/self.scale])
        target=np.eye(3)[y]
        shape=(z.shape[1],3)
        def objective(flat):
            w=flat.reshape(shape); p=softmax(z@w,axis=1)
            loss=-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()+.01*np.mean(w[1:]**2)
            grad=z.T@(p-target)/len(y)
            grad[1:]+=.02*w[1:]/w[1:].size
            return loss,grad.ravel()
        result=minimize(objective,np.zeros(np.prod(shape)),method='L-BFGS-B',jac=True,
                        options={'maxiter':150,'ftol':1e-8})
        if not result.success: raise RuntimeError('Shape optimizer failed: '+str(result.message))
        self.weights=result.x.reshape(shape); self.temperature=1.
        self.temperature=fit_temperature(self.predict(cal_x),cal_y)
        return self

    def predict(self,x):
        def raw(values):
            z=np.column_stack([np.ones(len(values)),(values-self.mean)/self.scale])
            return softmax(z@self.weights,axis=1)
        # Exact team-swap equivariance, including any optimizer round-off.
        p=(raw(x)+raw(swap_features(x))[:,[1,0,2]])/2
        return temperature(p,self.temperature)

    def save(self,path):
        np.savez_compressed(path,mean=self.mean,scale=self.scale,weights=self.weights,
                            temperature=self.temperature,team_dim=TEAM_DIM,stride=STRIDE)

    @classmethod
    def load(cls,path):
        model=cls()
        with np.load(path,allow_pickle=False) as z:
            if int(z['stride'])!=STRIDE or int(z['team_dim'])!=TEAM_DIM:
                raise ValueError('Incompatible shape schema')
            model.mean=z['mean']; model.scale=z['scale']; model.weights=z['weights']
            model.temperature=float(z['temperature'])
        return model
