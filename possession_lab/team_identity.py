"""Estimate team identity on live frames with the dead-ball path held fixed."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch import nn

from .be_ha_gate import visibility_gate
from .ensemble import decode
from .kinematics import velocity
from .moe import prepare
from .probabilistic import fit_temperature
from .stoppage_evidence import home_given_alive
from .team_shape import shape_features, TEAM_DIM, STRIDE
from .temporal import chunk_ranges

SEED = 7
EPOCHS = 6
RADIUS = 15
CHUNK = 512
WEIGHT = .5



def player_features(data):
    shape = shape_features(data).reshape(-1,2,TEAM_DIM)
    v = velocity(data.players, lag=5, maximum_speed=15., check_steps=True)[::STRIDE]
    features = []
    for team in (0,1):
        tv = v[:,data.teams==team]
        valid = np.isfinite(tv).all(axis=-1)
        n = valid.sum(axis=1)
        summ = np.nansum(tv,axis=1)
        speed = np.nansum(np.linalg.norm(tv,axis=-1),axis=1)
        collective = np.linalg.norm(summ,axis=-1)
        features.append(np.column_stack((n/11, speed/np.maximum(n,1)/15,
            collective/np.maximum(n,1)/15, collective/np.maximum(speed,1e-6))))
    return np.concatenate((shape,np.stack(features,axis=1)),axis=2).astype(np.float32)


class MotionHA(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Conv1d(width,24,1)
        self.layers = nn.ModuleList(nn.Conv1d(24,24,3,padding=d,dilation=d) for d in (1,2,4,8))
        self.head = nn.Conv1d(24,1,1)

    def forward(self, x):
        # Same scorer for both team blocks; swapping teams negates the log odds.
        b,t,_,d = x.shape
        h = self.project(x.permute(0,2,3,1).reshape(b*2,d,t)).relu()
        for layer in self.layers:
            h = (h+layer(h)).relu()
        score = self.head(h).reshape(b,2,t)
        return score[:,0] - score[:,1]


def motion_logits(model,x):
    output = []
    model.eval()
    with torch.inference_mode():
        for lo,hi,start,stop in chunk_ranges(len(x),1024,RADIUS):
            p = model(torch.from_numpy(x[None,lo:hi]))[0,start-lo:stop-lo]
            output.append(p.numpy())
    return np.concatenate(output)


def fit_specialists(halves,path):
    # Full temporal context stays within a physical half and one split.
    train,cal = [],[]
    for clean,noisy,y in halves:
        cut = int(.8*len(y))
        train.extend((x[:cut-25],y[:cut-25]) for x in (clean,noisy))
        cal.extend((x[cut+25:],y[cut+25:]) for x in (clean,noisy))
    valid = np.concatenate([y<2 for _,y in train])
    y = np.concatenate([y for _,y in train])[valid]
    x = np.concatenate([z[:,0,:TEAM_DIM]-z[:,1,:TEAM_DIM] for z,_ in train])[valid]
    scale_lin = np.maximum(np.sqrt((x*x).mean(0)),.02)
    x = (x/scale_lin).astype(np.float64)
    target = (y==0)
    def loss(w):
        z=x@w
        return (np.mean(np.logaddexp(0,z)-target*z)+.01*(w@w),
                x.T@(expit(z)-target)/len(z)+.02*w)
    fit = minimize(loss,np.zeros(TEAM_DIM),jac=True,method='L-BFGS-B')
    if not fit.success:
        raise RuntimeError(f'Team shape fit failed: {fit.message}')

    flat = np.concatenate([z for z,_ in train],axis=0).reshape(-1,train[0][0].shape[2])
    mean,scale = flat.mean(0),np.maximum(flat.std(0),.02)
    norm = lambda z: np.clip((z-mean)/scale,-10,10).astype(np.float32)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = MotionHA(len(mean))
    opt = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    portions = []
    for values,labels in train:
        x = norm(values)
        for lo,hi,start,stop in chunk_ranges(len(labels),CHUNK,RADIUS):
            yy = labels[start:stop]
            if (yy<2).any():
                portions.append((x[lo:hi],yy,start-lo))
    history = []
    for _ in range(EPOCHS):
        model.train()
        total,count = 0.,0
        opt.zero_grad(set_to_none=True)
        for batch, idx in enumerate(rng.permutation(len(portions)),1):
            x,y,left = portions[idx]
            active = y<2
            logits = model(torch.from_numpy(x[None]))[0,left:left+len(y)]
            value = nn.functional.binary_cross_entropy_with_logits(logits[active],
                torch.from_numpy((y[active]==0).astype(np.float32)),reduction='sum')
            (value/(CHUNK*4)).backward()
            if batch%4==0 or batch==len(portions):
                nn.utils.clip_grad_norm_(model.parameters(),5.)
                opt.step()
                opt.zero_grad(set_to_none=True)
            total += float(value.detach())
            count += int(active.sum())
        history.append(total/count)
    yc = np.concatenate([labels[labels<2] for _,labels in cal])
    movement = np.concatenate([motion_logits(model,norm(z))[labels<2] for z,labels in cal])
    linear = np.concatenate([((z[:,0,:TEAM_DIM]-z[:,1,:TEAM_DIM])/scale_lin@fit.x)[labels<2] for z,labels in cal])
    def calibrate(logits):
        p = expit(logits)
        return float(fit_temperature(np.column_stack((p,1-p)),yc))
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),mean=torch.from_numpy(mean),scale=torch.from_numpy(scale),
        motion_temp=calibrate(movement),linear_temp=calibrate(linear),scale_lin=torch.from_numpy(scale_lin),
        linear=torch.from_numpy(fit.x),training_nll=history),path)
    return Specialist(path)


class Specialist:
    def __init__(self,path):
        s = torch.load(path,map_location='cpu',weights_only=True)
        self.state = s
        self.model = MotionHA(len(s['mean']))
        self.model.load_state_dict(s['model'])

    def predict(self,x):
        s=self.state
        z=np.clip((x-s['mean'].numpy())/s['scale'].numpy(),-10,10).astype(np.float32)
        p=expit(motion_logits(self.model,z)/s['motion_temp'])
        q=expit((x[:,0,:TEAM_DIM]-x[:,1,:TEAM_DIM])/s['scale_lin'].numpy()@s['linear'].numpy()/s['linear_temp'])
        return q,p


def lock_dead(base,path,home,active,config,weight):
    """Re-decode only eligible live runs. Every D and every label outside the gate is pinned."""
    change = active & (path<2)
    if not change.any():
        return path.copy()
    trans = np.array(config['transition'])[:2,:2]
    trans = trans/trans.sum(1,keepdims=True)
    stay = (trans[0,0]+trans[1,1])/2
    logtrans = .25*np.log(np.array([[stay,1-stay],[1-stay,stay]]).clip(1e-9))
    anchor = base[:,0]/base[:,:2].sum(1).clip(1e-12)
    mix = (1-weight)*anchor+weight*home
    emission = np.log(np.column_stack((mix,1-mix)).clip(1e-9))
    result = path.copy()
    bounds = np.flatnonzero(np.diff(np.r_[False,change,False]))
    for lo,hi in bounds.reshape(-1,2):
        scores = emission[lo].copy()
        if lo and path[lo-1]<2:
            scores += logtrans[path[lo-1]]
        back = np.zeros((hi-lo,2),np.int8)
        for i in range(lo+1,hi):
            options = scores[:,None]+logtrans
            back[i-lo] = options.argmax(axis=0)
            scores = emission[i]+options.max(axis=0)
        if hi<len(path) and path[hi]<2:
            scores += logtrans[:,path[hi]]
        result[hi-1] = scores.argmax()
        for i in range(hi-2,lo-1,-1):
            result[i] = back[i+1-lo,result[i+1]]
    np.testing.assert_array_equal(result[~change],path[~change])
    np.testing.assert_array_equal(result==2,path==2)
    return result


def infer_without_ball(data,features,specialist,tcn,a,d,gate):
    e,q=prepare(data,tcn,a,d)
    base,_=gate.predict(e,q)
    _,path=decode(base,tcn.config)
    active,_,_=visibility_gate(data)
    ha_a=e[:,1,0]/e[:,1,:2].sum(1).clip(1e-12)
    shape,motion=specialist.predict(features)
    up = lambda p: np.interp(np.arange(0,len(data.ball),2),np.arange(0,len(data.ball),STRIDE),p)
    arms={'moe_d':path,'moe_d_crf_locked':lock_dead(base,path,ha_a,active,tcn.config,.35),
          'moe_d_shape':lock_dead(base,path,up(shape),active,tcn.config,.5),
          'moe_d_motion':lock_dead(base,path,up(motion),active,tcn.config,.5)}
    return {k:np.repeat(y,2)[:len(data.ball)] for k,y in arms.items()}, up(shape), up(motion)


def decode_p1_team(data, base_p, base_path, baseline_config, p1, observed=None):
    """Leave the original decisions in place where ball or team observations are absent."""
    finite = np.isfinite(data.players[::2]).all(axis=2)
    teams_visible = np.logical_and.reduce([
        finite[:,data.teams==team].sum(1)>=5 for team in (0,1)])
    active = np.isfinite(data.ball[::2]).all(1) & teams_visible & (base_path<2)
    if observed is not None:
        active &= observed
    if active.any():
        home = home_given_alive(p1.emissions(data))
        path = lock_dead(base_p,base_path,home,active,baseline_config,WEIGHT)
    else:
        path = base_path.copy()
    np.testing.assert_array_equal(path==2,base_path==2)
    np.testing.assert_array_equal(path[~active],base_path[~active])
    return path,active
