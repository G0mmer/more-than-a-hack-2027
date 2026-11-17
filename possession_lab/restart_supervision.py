"""Fit and apply the tracking-only stoppage and restart classifier heads."""
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.special import expit

from .stoppage_evidence import STEP, Logistic
from .stoppage_intervals import average_at, budgeted, candidates, fixed_scores

ARMS = ('moe_d','sparse_rule','dfl_restart','weak_free_kick','weak_restart')
HEADS = ('state','restart')
CONTROLLED = {'PASS','HIGH PASS','CROSS','DRIVE','HEADER','SHOT'}


def tracking_features(f):
    return np.column_stack((f['xm'],f['xs'],f['xh'][:,:27])).astype(np.float32)


def outside_events(actions,frames):
    events = [(f,a['label']) for f,a in zip(frames,actions)]
    pairs = list(zip(events,events[1:]))
    throws = [(a,b) for (a,x),(b,y) in pairs if x=='OUT' and y=='THROW IN' and 50<=b-a<=750]
    free = [b for (a,_),(b,y) in pairs if y=='FREE KICK' and 60<=b-a<=1125]
    live = [(a,b) for (a,x),(b,y) in pairs if x in CONTROLLED and y in CONTROLLED and 7<=b-a<=50]
    return throws,free,live


def weak_targets(actions,frames,n,include_throws):
    throws,free,live = outside_events(actions,frames)
    axis = np.arange(0,n,STEP)
    states = np.full(len(axis),-1,np.int8)
    restart = states.copy()
    edges = free+[b for _,b in throws]
    for lo,hi in live:
        states[(axis>=lo)&(axis<=hi)] = 0
        if all(abs(hi-b)>75 for b in edges):
            restart[abs(axis-hi)<=10] = 0
    pauses = [(b-38,b-8) for b in free]
    ends = list(free)
    if include_throws:
        pauses.extend((a+15,b-15) for a,b in throws)
        ends.extend(b for _,b in throws)
    for lo,hi in pauses:
        states[(axis>=lo)&(axis<hi)] = 1
    for b in ends:
        restart[abs(axis-b)<=10] = 1
    return dict(state=states,restart=restart),dict(out_then_throw=len(throws),free_kick=len(free),live_pairs=len(live))


def strong_targets(y):
    # Label a narrow neighborhood around verified D -> live boundaries.
    prev,curr = y[:-1],y[1:]
    starts = np.flatnonzero((prev==2)&(curr<2))+1
    relevant = np.zeros(len(y),dtype=np.int8)
    relevant[starts] = 1
    edge = maximum_filter1d(relevant,5,mode='constant')
    known = y!=3
    return dict(state=np.where(known,y==2,-1), restart=np.where(known,edge,-1))


def fit_heads(bank,sources,weak,weak_kind):
    rng = np.random.default_rng(711578)
    source = [item for m in sources for item in bank[m]]
    result = dict(sources=list(sources),weak_source='118578' if weak_kind else None,heads={},examples={})
    for head in HEADS:
        x = np.concatenate([a for a,_,_ in source])
        y = np.concatenate([b[head] for _,b,_ in source])
        cover = np.concatenate([c for _,_,c in source])
        selected = []
        weak_counts = [0,0]
        for label in (0,1):
            ids = np.flatnonzero((y==label)&cover)
            cap = min(16000 if head=='state' else 8000,len(ids))
            selected.append(rng.choice(ids,cap,replace=False))
        # Equalize class weight and keep weak examples near 20% of a class,
        # with a replication ceiling for scarce restart labels.
        n = min(map(len,selected))
        selected = [a[:n] for a in selected]
        rows = [x[a] for a in selected]
        target = [np.full(n,label) for label in (0,1)]
        if weak_kind:
            wx,wy,wcover = weak[weak_kind]
            pools = [np.flatnonzero((wy[head]==label)&wcover) for label in (0,1)]
            count = min(n//4,10*min(map(len,pools)))
            if not count:
                raise ValueError('No usable weak examples for '+head)
            for label,candidates_ in enumerate(pools):
                chosen = rng.choice(candidates_,count,replace=count>len(candidates_))
                rows.append(wx[chosen])
                target.append(np.full(count,label))
                weak_counts[label] = int(len(candidates_))
        joined = np.concatenate(rows)
        lab = np.concatenate(target)
        result['heads'][head] = Logistic.fit(joined,lab,l2=.012).as_dict()
        result['examples'][head] = dict(dfl_unique_per_class=n,weak_unique_available=weak_counts,
                                        fitted_rows=len(lab))
    return result


class Expert:
    def __init__(self,weights):
        self.heads = {k:Logistic(v) for k,v in weights['heads'].items()}

    def heads_at(self,f):
        x = tracking_features(f)
        return tuple(expit(self.heads[k].logits(x)) for k in HEADS)


def infer(f,base,experts):
    spans,_,policy = candidates(f)
    rule = fixed_scores(policy)
    sure = rule>=.49
    output = {'moe_d':base,'sparse_rule':budgeted(base,spans,sure,rule)}
    probabilities = {}
    left,right = spans.T
    for name,e in experts.items():
        state,restart = e.heads_at(f)
        probabilities[name] = restart
        csum = np.r_[0,np.cumsum(state)][:,None]
        mean = average_at(csum,left,right).ravel()
        before = average_at(csum,np.maximum(0,left-9),left).ravel()
        after = average_at(csum,right,np.minimum(len(state),right+9)).ravel()
        near_end = maximum_filter1d(restart,7,mode='nearest')[np.minimum(right,len(state)-1)]
        possible = ((left>=9)&(right+9<=len(state))&(policy[:,2]>=.8)&(mean>=.61)&
                    (mean-before>=.055)&(mean-after>=.055)&(near_end>=.61)&(right-left<=100))
        score = .58*mean+.42*near_end
        output[name] = budgeted(base,spans,sure|possible,np.where(sure,2+rule,score))
    return output,probabilities


def restart_ap(truth,score):
    known, relevant = truth>=0,truth==1
    y = relevant[known]
    ranked = np.argsort(-score[known],kind='stable')
    if not y.any():
        return None
    # Score must separate true restarts from ordinary live frames.
    flags = y[ranked]
    return float((np.cumsum(flags)[flags]/(1+np.flatnonzero(flags))).sum()/flags.sum())
