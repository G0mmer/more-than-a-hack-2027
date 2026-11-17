"""Bounded interval proposals and evidence composition for stoppage experiments."""
import numpy as np
from scipy.special import expit

from .evaluation import counts_f1
from .stoppage_evidence import STEP, Logistic, dead_metrics

DURATIONS = np.array([8, 14, 22, 34, 50, 70, 95, 130, 180, 225])
GRID = 8
INTERVAL_ARMS = ('moe_d', 'interval_rule', 'players', 'geometry', 'geometry_strict', 'local_moe')
WEIGHTED_ARMS = ('moe_d', 'sparse_rule', 'motion_boundaries', 'motion_ball', 'motion_ball_crf')
RANKED_ARMS = (*WEIGHTED_ARMS, 'ranked_motion_ball', 'ranked_motion_crf')


def average_at(csum, left, right):
    return (csum[right] - csum[left]) / np.maximum(right-left, 1)[:, None]


def candidates(f, players_only=False):
    """5 Hz candidates. All feature and proposal logic is label-free."""
    n = len(f['xm'])
    move, shape, ball = f['xm'], f['xs'], f['xb']
    seen, border, stuck = (f[k].astype(float) for k in ('seen', 'border', 'stuck'))
    valid = f['mv'].astype(float)
    outside = (border < -.30)*seen
    near = (np.abs(border) < 1.5)*seen
    player_series = np.column_stack((move[:, 1]*valid, move[:, 5]*valid,
        move[:, 2]*valid, move[:, 6]*valid, move[:, 0]*valid, move[:, 4]*valid,
        shape[:, 8]*f['sv'], np.abs(shape[:, 17])*f['sv'], valid, f['sv']))
    ball_series = np.column_stack((seen, outside, near, stuck, ball[:, 5]*seen,
        ball[:, 6]*seen, ball[:, 2]*seen, (ball[:, 1]<.10)*seen))
    series = player_series if players_only else np.column_stack((player_series, ball_series))
    csum = np.vstack([np.zeros(series.shape[1]), np.cumsum(series, axis=0, dtype=float)])
    idx = np.arange(n)
    pre = average_at(csum, np.maximum(idx-7, 0), idx)
    post = average_at(csum, idx, np.minimum(idx+7, n))
    onset = (pre[:, 3]-post[:, 3]) + .5*(post[:, 0]-pre[:, 0])
    restart = (post[:, 3]-pre[:, 3]) + .5*(pre[:, 0]-post[:, 0])
    if not players_only:
        onset += .65*(post[:, 11]-pre[:, 11]) + .25*pre[:,12]*(pre[:,10]-post[:,10]).clip(0)
        restart += .65*(pre[:, 11]-post[:, 11]) + .25*(post[:,10]-pre[:,10]).clip(0)
    grid = np.arange(0, n, GRID)
    peak = np.minimum(grid[:, None]+np.arange(GRID), n-1)
    starts = np.unique(np.r_[grid, peak[np.arange(len(peak)), np.argmax(onset[peak], axis=1)]])
    start = np.repeat(starts, len(DURATIONS))
    fixed_end = np.minimum(start+np.tile(DURATIONS, len(starts)), n)
    search = np.clip(fixed_end[:, None]+np.arange(-4,5), start[:,None]+1, n)
    # Give stopped play and actual restart evidence a chance to set the right edge.
    edge_value = np.r_[restart, 0.][search] - .025*abs(search-fixed_end[:,None])
    end = search[np.arange(len(search)), edge_value.argmax(axis=1)]
    spans = np.unique(np.column_stack((start,end)), axis=0)
    spans = spans[spans[:,1]-spans[:,0]>=7]
    left,right = spans.T
    duration = right-left
    middle = average_at(csum, left, right)
    before = average_at(csum, np.maximum(0,left-10), left)
    after = average_at(csum, right, np.minimum(n,right+10))
    policy = np.column_stack((
        middle[:,0], middle[:,1], middle[:,8],
        before[:,3]-middle[:,3], after[:,3]-middle[:,3],
        middle[:,0]-np.maximum(before[:,0],after[:,0]),
        middle[:,11] if not players_only else np.zeros(len(left)),
        middle[:,13] if not players_only else np.zeros(len(left))))
    x = np.column_stack((middle, before-middle, after-middle, np.log1p(duration/5),
        onset[left], np.r_[restart,0.][right])).astype(np.float32)
    return spans, x, policy


def target(spans, truth):
    """Candidate target based on at least 0.5 interval IoU with an actual stoppage."""
    dead = truth == 2
    edges = np.flatnonzero(np.diff(np.r_[False,dead,False]))
    left, right = spans.T
    best = np.zeros(len(spans))
    max_duration = (right-left).max(initial=0)
    for a,b in edges.reshape(-1,2):
        lo,hi = np.searchsorted(left,(a-max_duration+1,b))
        l,r = left[lo:hi],right[lo:hi]
        overlap = np.maximum(0,np.minimum(b,r)-np.maximum(a,l))
        iou = overlap/(r-l+b-a-overlap)
        best[lo:hi] = np.maximum(best[lo:hi],iou)
    return best >= .50


def fixed_scores(policy):
    lo_still,hi_still,coverage,slowdown,speedup,still_delta,outside,stuck = policy.T
    exit_field = outside >= .38
    formation = ((coverage >= .85) & (lo_still >= .32) &
                 (slowdown >= .10) & (speedup >= .10) & (still_delta >= .08))
    # Stationary ball alone is weak evidence; players still need to support the rule.
    exit_score = np.where(exit_field, .85*outside+.20*lo_still+.10*hi_still, 0.)
    formation_score = np.where(formation, .24+slowdown+speedup+.3*still_delta+.1*stuck, 0.)
    return np.maximum(exit_score,formation_score)


def decode_spans(spans, scores, threshold, base):
    """Add non-overlapping high-quality intervals while keeping existing D and H/A."""
    n = len(base)
    left,right = (spans*STEP).T
    right = np.minimum(right,n)
    active = (base!=2).astype(np.int32)
    cum = np.r_[0,np.cumsum(active)]
    changed = cum[right]-cum[left]
    chosen = np.flatnonzero((scores >= threshold)&(changed >= 25))
    # A broad high score is preferred to several marginal adjacent short calls.
    priority = (scores[chosen]-threshold)*np.sqrt(changed[chosen]/25)
    order = chosen[np.argsort(-priority, kind='stable')]
    occupied = np.zeros(n,dtype=bool)
    path = base.copy()
    for i in order:
        start, stop = left[i],right[i]
        buffer = slice(max(0,start-20),min(n,stop+20))
        if not occupied[buffer].any():
            occupied[start:stop] = True
            path[start:stop] = 2
    return path


def local_expert_boost(spans, post, f):
    # Disagreement isn't calibrated certainty. Bound the added evidence and require
    # persistence for the specialist signal so short CRF spikes have limited weight.
    left,right = spans.T
    reliable_ball = f['seen']*(~(f['stuck']>.5))
    signals = np.column_stack((
        (post[:,2]>.70).astype(float), (post[:,3]>.70).astype(float),
        ((post[:,1]>.65)&(post[:,3]>.65)).astype(float),
        reliable_ball*(1-post[:,0])))
    csum = np.vstack([np.zeros(signals.shape[1]), np.cumsum(signals,axis=0)])
    motion, specialist, consensus, conflict = average_at(csum,left,right).T
    return .65*motion + .85*specialist + .45*consensus - .35*conflict


def infer_intervals(f, post, base, weights):
    paths = {'moe_d':base}
    primary = None
    for arm in ('players','geometry'):
        spans,x,policy = candidates(f,players_only=arm=='players')
        logits = Logistic(weights[arm]).logits(x)
        paths[arm] = decode_spans(spans,expit(logits),.40,base)
        if arm=='geometry':
            primary = spans,policy,logits
    spans,policy,logits = primary
    paths['geometry_strict'] = decode_spans(spans,expit(logits),.58,base)
    paths['interval_rule'] = decode_spans(spans,fixed_scores(policy),.49,base)
    mixed = logits + local_expert_boost(spans,post,f)
    paths['local_moe'] = decode_spans(spans,expit(mixed),.50,base)
    return paths


def budgeted(base, spans, allowed, priority):
    n = len(base)
    left,right = (spans*STEP).T
    right = np.minimum(right,n)
    padding = 50
    protected = np.r_[0,np.cumsum(base==2)]
    nearby = protected[np.minimum(n,right+padding)]-protected[np.maximum(0,left-padding)]
    size = right-left
    permitted = np.flatnonzero(allowed&(nearby==0)&(size>=75)&(size<=575))
    ranked = permitted[np.argsort(-priority[permitted],kind='stable')]
    budget = max(250, int(n*.015))
    changed = np.zeros(n,dtype=bool)
    seen_windows = set()
    for i in ranked:
        start, stop = left[i],right[i]
        cell = ((start+stop)//2)//(360*25)
        if size[i]>budget or cell in seen_windows:
            continue
        if not changed[max(0,start-padding):min(n,stop+padding)].any():
            changed[start:stop] = True
            budget -= size[i]
            seen_windows.add(cell)
    return np.where(changed,2,base)


def guarded(f, post, base, weights):
    spans,x,policy = candidates(f)
    scores = fixed_scores(policy)
    slow, _, cover, brake, resume, extra_still, outside, _ = policy.T
    motion = ((cover>=.80)&(slow>=.28)&(brake>=.08)&(resume>=.08)&(extra_still>=.05))
    ball = outside>=.40
    votes = local_expert_boost(spans,post,f)
    p = expit(Logistic(weights['geometry']).logits(x)+votes)
    return {
        'sparse_rule': budgeted(base,spans,scores>=.49,scores),
        'sparse_moe': budgeted(base,spans,(p>=.52)&(motion|ball|(votes>=.60)),p),
        'sparse_moe_strict': budgeted(base,spans,(p>=.63)&(motion|ball),p),
    }


def roll(values):
    values = np.column_stack(values)
    return np.vstack([np.zeros(values.shape[1]),np.cumsum(values,axis=0,dtype=float)])


def movement_features(f,spans,geometry):
    """Use the player-only model on exactly the same intervals as the other signals."""
    # The geometry vector stores three 18-column aggregates. The first ten in
    # each aggregate are the player-only columns. Boundary cues are recomputed
    # without ball observations, as in the original player's training features.
    n = len(f['xm'])
    idx = np.arange(n+1)
    cover = f['mv']
    csum = roll((f['xm'][:,1]*cover,f['xm'][:,6]*cover))
    before = average_at(csum,np.maximum(0,idx-7),idx)
    after = average_at(csum,idx,np.minimum(n,idx+7))
    shift = (before[:,1]-after[:,1]) + .5*(after[:,0]-before[:,0])
    shift[n] = 0.
    left,right = spans.T
    return np.column_stack((geometry[:,:10],geometry[:,18:28],geometry[:,36:46],
                            geometry[:,54],shift[left],-shift[right])).astype(np.float32)


def context_scores(f,post,spans):
    left,right = spans.T
    seen = f['seen'].astype(float)
    outside = seen*(f['border']<-.30)
    near = seen*(abs(f['border'])<1.5)
    csum = roll((seen,outside,near,(post[:,2]>.70),(post[:,3]>.70)))
    inside = average_at(csum,left,right)
    before = average_at(csum,np.maximum(left-8,0),left)
    sight_lost = (before[:,0]-inside[:,0]).clip(0,1)
    ball = (1.5*inside[:,1]+.18*inside[:,2]+.48*before[:,2]*sight_lost).clip(0,1)
    crf = .35*inside[:,3]+.65*inside[:,4]
    return ball,crf,inside[:,0]


def refine(base,seed,spans,rule,motion,coverage):
    """Extend a rule interval only with stronger evidence for its wider boundaries."""
    out = seed.copy()
    bounds = np.flatnonzero(np.diff(np.r_[False,(seed==2)&(base!=2),False]))
    left,right = (spans*STEP).T
    right = np.minimum(right,len(out))
    budget = max(0,max(250,int(len(out)*.015))-int((out!=base).sum()))
    for a,b in bounds.reshape(-1,2):
        same = (left==a)&(right==b)
        if not same.any():
            raise ValueError('Rule interval is missing from the common proposal grid')
        p0 = motion[same].max()
        r0 = rule[same].max()
        added = (a-left)+(right-b)
        allowed = ((left<=a)&(left>=a-80)&(right>=b)&(right<=b+80)&
                   (added>=25)&(added<=min(150,budget))&(right-left<=725)&
                   (motion>=max(.23,p0+.035))&(coverage>=.8))
        score = .68*(motion-p0)+.32*(rule-r0)
        eligible = np.flatnonzero(allowed&(score>.012))
        for i in eligible[np.argsort(-score[eligible],kind='stable')]:
            lo,hi = left[i],right[i]
            pad_before = out[max(0,lo-50):a]
            pad_after = out[b:min(len(out),hi+50)]
            if (pad_before!=2).all() and (pad_after!=2).all():
                out[lo:hi] = 2
                budget -= added[i]
                break
    return out


def fuse(f,post,base,weights):
    spans,geometry,policy = candidates(f)
    rule = fixed_scores(policy)
    coverage = policy[:,2]
    motion = expit(Logistic(weights['players']).logits(movement_features(f,spans,geometry)))
    ball,crf,seen = context_scores(f,post,spans)
    seed = budgeted(base,spans,rule>=.49,rule)
    # Preserve rule choices within the same budget. Rank additional intervals
    # only when a separate signal corroborates the player-only classifier.
    physical = .62*motion+.38*ball
    with_experts = np.where(seen>=.30,.56*motion+.29*ball+.15*crf,.68*motion+.32*crf)
    can_add = (coverage>=.80)&(motion>=.28)
    extra_ball = can_add&(ball>=.26)&(physical>=.39)
    extra_crf = can_add&((ball>=.26)|(crf>=.60))&(with_experts>=.40)
    return {
        'moe_d':base,
        'sparse_rule':seed,
        'motion_boundaries':refine(base,seed,spans,rule,motion,coverage),
        'motion_ball':budgeted(base,spans,(rule>=.49)|extra_ball,np.where(rule>=.49,2+rule,physical)),
        'motion_ball_crf':budgeted(base,spans,(rule>=.49)|extra_crf,np.where(rule>=.49,2+rule,with_experts)),
    }


def add_ranked(base,seed,spans,coverage,motion,ball,crf,include_crf):
    n = len(base)
    left,right = (spans*STEP).T
    right = np.minimum(right,n)
    size = right-left
    protected = np.r_[0,np.cumsum(seed==2)]
    padding = 50
    neighborhood = protected[np.minimum(n,right+padding)]-protected[np.maximum(0,left-padding)]
    pool = (neighborhood==0)&(size>=75)&(size<=300)&(coverage>=.8)
    path = seed.copy()
    if not pool.any():
        return path

    # Relative movement strength compensates for motion differences across
    # tracking feeds. An absolute floor keeps empty or uniformly weak periods quiet.
    floor = max(.12,float(np.quantile(motion[pool],.95)))
    movement = np.minimum(motion/floor,2.)/2
    physical = (ball/.22).clip(0,1)
    corroboration = np.maximum(physical,.90*(crf/.32).clip(0,1)) if include_crf else physical
    same_window = (ball>=.10)|(crf>=.18) if include_crf else ball>=.10
    eligible = np.flatnonzero(pool&(motion>=floor)&same_window)
    score = .66*movement+.34*corroboration
    ranked = eligible[np.argsort(-score[eligible],kind='stable')]
    already_added = int((seed!=base).sum())
    budget = min(max(0,max(250,int(n*.015))-already_added),max(125,int(n*.007)))
    occupied_blocks = set()
    for i in ranked:
        a,b = left[i],right[i]
        block = (a+b)//(2*25*300)
        if size[i]>budget or block in occupied_blocks:
            continue
        if np.all(path[max(0,a-padding):min(n,b+padding)]!=2):
            path[a:b] = 2
            budget -= size[i]
            occupied_blocks.add(block)
    return path


def infer_ranked(f,post,base,weights):
    paths = fuse(f,post,base,weights)
    spans,geometry,policy = candidates(f)
    motion = expit(Logistic(weights['players']).logits(movement_features(f,spans,geometry)))
    ball,crf,_ = context_scores(f,post,spans)
    seed = paths['sparse_rule']
    for arm,include_crf in zip(RANKED_ARMS[-2:],(False,True)):
        paths[arm] = add_ranked(base,seed,spans,policy[:,2],motion,ball,crf,include_crf)
    return paths


def metrics(pm, periods, paths):
    result = dead_metrics(pm, periods, paths)
    n_match = n_gt = n_pred = overlap = 0
    for (_,y),z in zip(periods,paths):
        for lo,hi in pm.valid_runs(y):
            g,p = [[(a,b,k) for a,b,k in pm.segments(arr,[(lo,hi)]) if k==pm.D]
                   for arr in (y,z)]
            n_gt += len(g)
            n_pred += len(p)
            n_match += pm.match_spells(pm.spell_overlaps(g,p),len(g),len(p),.3)[0]
            stop = np.array([b for a,b,k in p], dtype=int)
            for a,b,_ in g:
                i = np.searchsorted(stop,a,side='right')
                overlap += int(i<len(p) and p[i][0]<b)
    result.update(episode_recall=n_match/max(n_gt,1), episode_f1=counts_f1(n_match,n_pred-n_match,n_gt-n_match),
                  overlap_recall=overlap/max(n_gt,1), episodes=dict(iou30_matches=n_match,overlap_matches=overlap,
                  truth=n_gt,predicted=n_pred))
    return result
