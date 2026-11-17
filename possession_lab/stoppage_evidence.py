"""Shared tracking features and temporal evidence for stoppage experiments."""
from dataclasses import replace

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize
from scipy.special import expit, logit

from .ensemble import decode
from .evaluation import counts_f1
from .kinematics import velocity
from .moe import prepare
from .motion_tcn_metrics import diagnostics
from .robust import mask_ball
from .stress import perturb

STEP = 5
PURGE = 25
SCENARIOS = ('clean', 'long40', 'held_ball40', 'no_ball', 'no_ball_noise')
EVIDENCE_SCENARIOS = ('clean', 'long40', 'noise', 'no_ball', 'no_ball_noise')
EVIDENCE_ARMS = ('moe_d', 'rule', 'state', 'hazard')


def treatment(d,scenario,seed):
    if scenario in ('noise','no_ball_noise'):
        d=perturb(d,'noise_0.3m',seed)[0]
    if scenario=='long40':
        return mask_ball(d,.4,seed,True)[0]
    if scenario in ('no_ball','no_ball_noise'):
        return replace(d,ball=np.full_like(d.ball,np.nan))
    return d


def series_mean(x,n):
    return uniform_filter1d(x,n,axis=0,mode='nearest')


def difference(x):
    idx=np.arange(len(x))
    smooth=series_mean(x,9)
    return smooth[np.minimum(idx+8,len(x)-1)]-smooth[np.maximum(idx-8,0)]


def mean_valid(x,valid):
    return np.where(valid,x,0.).sum(1)/np.maximum(valid.sum(1),1)


def symmetric(a,b):
    return np.column_stack((np.minimum(a,b),np.maximum(a,b)))


def player_evidence(data):
    xy=np.asarray(data.players[::STEP],float)
    good=np.isfinite(xy).all(2)
    vel=velocity(xy,fps=5,lag=5,maximum_speed=15.,check_steps=True)
    speed=np.linalg.norm(vel,axis=2)
    valid=np.isfinite(speed)
    movement,shape,coverage,centres=[],[],[],[]
    for t in (0,1):
        own=data.teams==t
        s,v=speed[:,own],valid[:,own]
        v_n=v.sum(1)
        movement.append(np.column_stack((
            mean_valid(s,v)/8, ((s<.75)&v).sum(1)/np.maximum(v_n,1),
            ((s>3.)&v).sum(1)/np.maximum(v_n,1),
            ((s>5.)&v).sum(1)/np.maximum(v_n,1))))
        cover=good[:,own]
        points=xy[:,own]
        centre=np.stack([mean_valid(points[:,:,i],cover) for i in (0,1)],axis=1)
        std=np.stack([np.sqrt(mean_valid((points[:,:,i]-centre[:,i,None])**2,cover))
                      for i in (0,1)],axis=1)
        centre_ok=cover.sum(1)>=5
        coverage.append((v_n>=5,centre_ok))
        centres.append(centre)
        shape.append(np.column_stack((np.abs(centre/np.asarray(data.pitch)-.5),
                                      std/np.asarray(data.pitch))))
    m=symmetric(*movement)
    q=symmetric(*shape)
    inter=np.linalg.norm(centres[0]-centres[1],axis=1)/50
    q=np.column_stack((q,inter))
    mv=(series_mean((coverage[0][0]&coverage[1][0]).astype(float),21)>.7)
    sv=(series_mean((coverage[0][1]&coverage[1][1]).astype(float),21)>.7)
    # Difference direction is kept for hazards and made absolute for state recognition.
    xm=np.column_stack((series_mean(m,5),series_mean(m,15),np.abs(difference(m))))
    xs=np.column_stack((series_mean(q,9),np.abs(difference(q))))
    delta=np.column_stack((difference(m),difference(q),series_mean(m,5),mv,sv))
    delta[~mv,:m.shape[1]]=0
    delta[~sv,m.shape[1]:m.shape[1]+q.shape[1]]=0
    return xm,xs,delta,mv,sv,xy,vel


def evidence(data, players=None):
    if players is None:
        players=player_evidence(data)
    xm,xs,delta,mv,sv,xy,v=players
    ball=np.asarray(data.ball[::STEP],float)
    seen=np.isfinite(ball).all(1)
    speed=np.linalg.norm(velocity(ball,fps=5,lag=1,maximum_speed=60.,check_steps=True),axis=1)
    speed_ok=np.isfinite(speed)
    border=np.minimum(ball,np.asarray(data.pitch)-ball).min(axis=1)
    offset=xy-ball[:,None,:]
    dist=np.linalg.norm(offset,axis=2)
    nearest=np.where(np.isfinite(dist),dist,30.).min(1).clip(0,30)
    # Radial player velocity relative to the observed ball location.
    radial=np.sum(v*offset,axis=2)/np.maximum(dist,.3)
    near=np.isfinite(radial)&(dist<8)
    away=((radial>.7)&near).sum(1)/np.maximum(near.sum(1),1)
    approach=((radial<-.7)&near).sum(1)/np.maximum(near.sum(1),1)
    xb=np.column_stack((
        np.clip(np.nan_to_num(border,nan=0.),-8,15)/10,
        nearest/30.,np.nan_to_num(speed,nan=0.)/30,
        series_mean(((speed<.75)&speed_ok).astype(float),9),
        series_mean(((border<-.3)&seen).astype(float),5),away,approach,
        difference(approach),difference(away),speed_ok))
    bv=seen & (series_mean(seen.astype(float),9)>=.5)
    # Two types of transition, symmetric in the teams and independent of H/A identity.
    xh=np.column_stack((delta,np.where(bv[:,None],xb,0.),bv))
    return dict(xm=xm,xs=xs,xb=xb,xh=xh,mv=mv,sv=sv,bv=bv)


class Logistic:
    def __init__(self,record):
        self.mean,self.scale,self.beta=(np.asarray(record[k],float) for k in ('mean','scale','beta'))

    @classmethod
    def fit(cls,x,y,l2=.004,positive_weights=False):
        y=np.asarray(y,float)
        if len(y)==0 or len(np.unique(y))!=2:
            raise ValueError('Both binary states are required')
        mean,scale=x.mean(0),np.maximum(x.std(0),.02)
        z=np.column_stack((np.ones(len(x)),np.clip((x-mean)/scale,-10,10)))
        def objective(w):
            a=z@w
            loss=np.mean(np.logaddexp(0,a)-y*a)+l2*np.sum(w[1:]**2)/2
            gradient=z.T@(expit(a)-y)/len(y)
            gradient[1:]+=l2*w[1:]
            return loss,gradient
        init=np.zeros(z.shape[1]);init[0]=logit(y.mean())
        bounds=([(None,None)]+[(0,None)]*(z.shape[1]-1)) if positive_weights else None
        result=minimize(objective,init,method='L-BFGS-B',jac=True,bounds=bounds,
                        options={'maxiter':140,'ftol':1e-9})
        if not result.success:
            raise ValueError('Logistic fit failed: '+str(result.message))
        return cls(dict(mean=mean,scale=scale,beta=result.x))

    def logits(self,x):
        return self.beta[0]+np.clip((x-self.mean)/self.scale,-10,10)@self.beta[1:]

    def as_dict(self):
        return dict(mean=self.mean.tolist(),scale=self.scale.tolist(),beta=self.beta.tolist())


def split_view(f,y,lo,hi):
    return {k:v[lo:hi] for k,v in f.items()},y[lo:hi]


def group_logits(models,priors,f):
    return np.column_stack([np.where(f[k+'v'],np.clip(models[k].logits(f['x'+k])-priors[k],-8,8),0.)
                            for k in ('m','s','b')])


def feature_bank(data, match_order):
    bank={}
    for m,halves in data.items():
        for i,(d,y) in enumerate(halves):
            seed=1000+10*match_order.index(m)+i
            views={}
            clean=player_evidence(d)
            noisy=treatment(d,'noise',seed)
            noise=player_evidence(noisy)
            for name in EVIDENCE_SCENARIOS:
                changed=treatment(d,name,seed)
                views[name]=(evidence(changed,noise if 'noise' in name else clean),y[::STEP]==2)
            bank[m,i]=views
        print('Prepared evidence',m,flush=True)
    return bank


def fit_evidence(bank,sources):
    train,tail=[],[]
    for (m,_),views in bank.items():
        if m not in sources:
            continue
        for name,(f,y) in views.items():
            cut=int(.8*len(y))
            # All windows and labels touching the boundary are set aside.
            if name in ('clean','noise','long40'):
                train.append(split_view(f,y,0,cut-PURGE))
            tail.append(split_view(f,y,cut+PURGE,len(y)))
    models,priors={},{}
    for k in ('m','s','b'):
        x=np.concatenate([f['x'+k][f[k+'v']] for f,_ in train])
        y=np.concatenate([y[f[k+'v']] for f,y in train])
        models[k]=Logistic.fit(x,y)
        priors[k]=float(logit(y.mean()))
    sx=np.concatenate([group_logits(models,priors,f) for f,_ in tail])
    sy=np.concatenate([y for _,y in tail])
    stack=Logistic.fit(sx,sy,l2=.02,positive_weights=True)
    # Conditional hazards given the state at the previous sample. Noise and missingness
    # alter evidence only; all perturbed views reuse the source state sequence.
    hx=[f['xh'][1:] for f,_ in train+tail]
    hp=[y[:-1] for _,y in train+tail]
    hy=[y[1:] for _,y in train+tail]
    transitions={}
    for state,name in ((False,'onset'),(True,'restart')):
        lab=np.concatenate([y[p==state]!=state for p,y in zip(hp,hy)])
        # One view per source-tail condition is not required for temperature here:
        # likelihood penalty is fixed and whole evaluated matches are excluded.
        x=np.concatenate([v[p==state] for p,v in zip(hp,hx)])
        transitions[name]=dict(model=Logistic.fit(x,lab,l2=.008).as_dict(),rate=float(lab.mean()))
    return dict(models={k:v.as_dict() for k,v in models.items()}, priors=priors,
                stack=stack.as_dict(),prior=float(sy.mean()),transitions=transitions,sources=list(sources))


class EvidenceModel:
    def __init__(self,record):
        self.prior=record['prior']
        self.model={k:Logistic(v) for k,v in record['models'].items()}
        self.priors=record['priors']
        self.stack=Logistic(record['stack'])
        self.hazards={k:Logistic(v['model']) for k,v in record['transitions'].items()}
        self.rates={k:v['rate'] for k,v in record['transitions'].items()}

    def state_log_ratio(self,f):
        return np.clip(self.stack.logits(group_logits(self.model,self.priors,f))-logit(self.prior),-7,7)

    def transitions(self,f,dynamic):
        n=len(f['xh']); rates=[]
        for name,rate in self.rates.items():
            if dynamic:
                raw=np.clip(self.hazards[name].logits(f['xh'])-logit(rate),-3,3)
                raw=np.where(f['mv']|f['bv'],raw,0.)
                rate=expit(logit(rate)+.70*raw)
            rates.append(np.broadcast_to(rate,n))
        return rates


def viterbi(log_ratio,start_p,stop_p,prior):
    """Two-state Bayesian MAP path with optional input-dependent transitions."""
    probs=np.stack((np.column_stack((1-start_p,start_p)),np.column_stack((stop_p,1-stop_p))),axis=1)
    transitions=np.log(probs.clip(1e-8,1.))
    score=np.log([1-prior,prior])+np.array([0.,log_ratio[0]])
    back=np.zeros((len(log_ratio),2),np.int8)
    for i in range(1,len(log_ratio)):
        candidates=score[:,None]+transitions[i]
        back[i]=candidates.argmax(axis=0)
        score=candidates.max(axis=0)+np.array([0.,log_ratio[i]])
    path=np.empty(len(log_ratio),np.int8);path[-1]=score.argmax()
    for i in range(len(path)-2,-1,-1):
        path[i]=back[i+1,path[i+1]]
    return path.astype(bool)


def rule_ratio(f):
    # Fixed, interpretable support in log-odds units.
    fraction_still=f['xm'][:,1]  # lower still fraction for the two teams
    fast_delta=f['xh'][:,6]      # larger of the two brisk fractions, post minus pre
    slow=np.where(f['mv']&(fraction_still>.45),1.3,0.)
    stopped=np.where(f['mv']&(fast_delta<-.15),1.2,0.)
    restart=np.where(f['mv']&(fast_delta>.15),-1.4,0.)
    outside=np.where(f['bv']&(f['xb'][:,4]>.60),2.2,0.)
    return slow+stopped+restart+outside


def infer_evidence(data,f,model,tcn,a,d,gate):
    experts,quality=prepare(data,tcn,a,d)
    p,_=gate.predict(experts,quality)
    _,anchor=decode(p,tcn.config)
    n=len(data.ball)
    base=np.repeat(anchor,2)[:n]
    sample=np.arange(0,n,STEP)
    to5=lambda x:np.interp(sample,np.arange(0,n,2),x)
    ball_weight=np.where(f['bv'],.80,.45)
    z=logit(to5(p[:,2]).clip(1e-5,1-1e-5))-logit(model.prior)
    pstate=model.state_log_ratio(f)
    available=f['mv']|f['sv']|f['bv']
    team=np.repeat((home_given_alive(p)<.5).astype(np.int8),2)[:n]
    result={'moe_d':base}
    for arm in EVIDENCE_ARMS[1:]:
        if arm=='rule':
            log_ratio=z+rule_ratio(f)
        else:
            log_ratio=np.where(available,ball_weight*z+(1-ball_weight)*pstate,z)
        on,off=model.transitions(f,arm=='hazard')
        is_d=np.repeat(viterbi(.65*log_ratio,on,off,model.prior),STEP)[:n]
        is_d=np.where(np.repeat(available,STEP)[:n],is_d,base==2)
        result[arm]=np.where(is_d,2,np.where(base<2,base,team)).astype(np.int8)
    return result


def dead_metrics(pm,data,predictions):
    # Frame metrics and signed boundary types. Count false D time directly.
    output=diagnostics(pm,data,predictions)
    starts=[0,0,0];restarts=[0,0,0]
    for (_,y),pred in zip(data,predictions):
        for run in pm.valid_runs(y):
            transitions=[pm.transitions(z,[run]) for z in (y,pred)]
            for entering,total in ((True,starts),(False,restarts)):
                events=[[(i,0,1) for i,a,b in s if (a==2)!=(b==2) and (b==2)==entering]
                        for s in transitions]
                pair,miss,extra=pm.match_transitions(*events,25,typed='pair')
                total[0]+=len(pair);total[1]+=len(extra);total[2]+=len(miss)
    truth=np.concatenate([y for _,y in data]);pred=np.concatenate(predictions)
    gt=truth==2; pd=pred==2
    output.update(d_recall=float((gt&pd).sum()/max(1,gt.sum())),
        false_d_per_live_minute=float(60*(pd&(truth<2)).sum()/max(1,(truth<2).sum())),
        onset_f1=counts_f1(*starts),onset_recall=starts[0]/max(1,starts[0]+starts[2]),
        resume_f1=counts_f1(*restarts),resume_recall=restarts[0]/max(1,restarts[0]+restarts[2]),
        onset_counts=starts,resume_counts=restarts)
    return output


def home_given_alive(p):
    live = p[:,:2].sum(axis=1)
    return np.divide(p[:,0],live,out=np.full(len(p),.5),where=live>0)


def corrupt(data, scenario, seed):
    if scenario != 'held_ball40':
        return treatment(data, scenario, seed)
    masked = treatment(data, 'long40', seed)
    originally_seen = np.isfinite(data.ball).all(axis=1)
    affected = originally_seen & ~np.isfinite(masked.ball).all(axis=1)
    visible = originally_seen & ~affected
    last_seen = np.maximum.accumulate(np.where(visible, np.arange(len(visible)), -1))
    use = affected & (last_seen >= 0)
    ball = masked.ball.copy()
    # A tracking failure can leave an apparently valid but frozen in-bounds ball.
    # Extend only actual injected gaps, using only positions observed before them.
    ball[use] = np.clip(data.ball[last_seen[use]], .4, np.asarray(data.pitch)-.4)
    return replace(data, ball=ball)


def emission_pack(data, tcn, a, d, gate):
    expert, features = prepare(data, tcn, a, d)
    p, _ = gate.predict(expert, features)
    _, states = decode(p, tcn.config)
    n = len(data.ball)
    src = np.arange(0, n, 2)
    sample = np.arange(0, n, STEP)
    values = np.column_stack((p[:, 2], expert[:, :, 2]))
    return np.repeat(states, 2)[:n], np.column_stack([
        np.interp(sample, src, col) for col in values.T]).astype(np.float32)


def auxiliary(data):
    ball = np.asarray(data.ball[::STEP], float)
    seen = np.isfinite(ball).all(axis=1)
    border = np.clip(np.minimum(ball, np.asarray(data.pitch)-ball).min(axis=1), -8, 16)
    changed = np.linalg.norm(np.diff(ball, axis=0, prepend=ball[:1]), axis=1)
    # Float tolerances are in metres. This only describes an observation pattern.
    stuck = seen & np.r_[False, seen[:-1]] & (changed < .02)
    return dict(seen=seen, border=np.where(seen, border, 0.).astype(np.float32), stuck=stuck)
