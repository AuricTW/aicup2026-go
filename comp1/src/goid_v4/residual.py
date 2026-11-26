"""P3 fixed semantic phi and exact legal-action expectation, no external teacher."""
from __future__ import annotations
import math
import numpy as np
from .artifacts import ContractError

PHI_NAMES=('row_fraction','column_fraction','edge_fraction','corner_fraction',
           'adjacent_empty_fraction','adjacent_own_fraction','adjacent_enemy_fraction',
           *[f'adjacent_friendly_liberty_bucket_{i}' for i in range(6)],
           'previous_distance_fraction','previous_available','is_pass')


def semantic_phi(decision):
    """16D descriptors; liberty is maximum adjacent friendly PRE-state group liberty.

    Bucket0 means no adjacent friendly group; buckets1..4 exact,5 >=5.
    Pass uses its own is_pass vector. No post-move outcome/value is invented.
    """
    b=decision.board;n=len(b);own=1 if decision.target_color=='B' else -1
    phi=np.zeros((n*n+1,len(PHI_NAMES)),dtype=np.float64)
    neighbours=lambda r,c:[(rr,cc) for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)) if 0<=rr<n and 0<=cc<n]
    group_lib={};seen=set()
    for r in range(n):
        for c in range(n):
            if b[r,c]!=own or (r,c) in seen:continue
            stack=[(r,c)];group={(r,c)};lib=set()
            while stack:
                rr,cc=stack.pop()
                for point in neighbours(rr,cc):
                    if b[point]==0:lib.add(point)
                    elif b[point]==own and point not in group:group.add(point);stack.append(point)
            seen.update(group)
            for point in group:group_lib[point]=len(lib)
    for r in range(n):
        for c in range(n):
            i=r*n+c;adj=neighbours(r,c)
            edge=min(r,c,n-1-r,n-1-c)
            corner=min(max(abs(r-rr),abs(c-cc)) for rr,cc in ((0,0),(0,n-1),(n-1,0),(n-1,n-1)))
            phi[i,:7]=(r/(n-1),c/(n-1),edge/max(1,n//2),corner/(n-1),
                       sum(b[p]==0 for p in adj)/4,sum(b[p]==own for p in adj)/4,sum(b[p]==-own for p in adj)/4)
            lib=max((group_lib.get(p,0) for p in adj),default=0)
            phi[i,7+min(lib,5)]=1
            if decision.previous_action is not None:
                pr,pc=divmod(decision.previous_action,n)
                phi[i,13]=(abs(r-pr)+abs(c-pc))/(2*(n-1));phi[i,14]=1
    phi[-1,15]=1
    return phi


def extract_residual(logits,decision,*,phi=None):
    logits=np.asarray(logits,dtype=np.float64)
    legal=decision.legal_mask
    if legal is None or legal.dtype!=bool or logits.shape!=legal.shape or not np.isfinite(logits).all() or not legal[-1] or not legal[decision.action]:
        raise ContractError("Policy/legal/actual-action contract failed")
    phi=semantic_phi(decision) if phi is None else np.asarray(phi,dtype=np.float64)
    if phi.ndim!=2 or phi.shape[0]!=len(logits) or not np.isfinite(phi).all():
        raise ContractError("Shared semantic phi mismatch")
    z=logits[legal];maximum=float(z.max())
    lognorm=maximum+math.log(float(np.exp(z-maximum).sum()))
    logp=np.full(len(logits),-np.inf);logp[legal]=z-lognorm
    probs=np.zeros(len(logits));probs[legal]=np.exp(logp[legal])
    if not np.isclose(probs.sum(),1,atol=1e-12):raise ContractError("Policy normalization failed")
    expected=probs@phi
    actual=decision.action
    order=np.flatnonzero(legal)[np.argsort(-logits[legal],kind='stable')]
    rank=int(np.flatnonzero(order==actual)[0])+1
    return {'raw':phi[actual].copy(),'residual':phi[actual]-expected,'expectation':expected,
            'log_probability':float(logp[actual]),'entropy':float(-(probs[legal]*logp[legal]).sum()),
            'policy_rank':rank,'probabilities':probs}


def teacher_quality_gate(teacher_nll,uniform_nll,frequency_nll):
    arrays=[np.asarray(x,dtype=np.float64) for x in (teacher_nll,uniform_nll,frequency_nll)]
    if arrays[0].ndim!=1 or len(arrays[0])==0 or any(x.shape!=arrays[0].shape or not np.isfinite(x).all() or (x<0).any() for x in arrays):
        raise ContractError("Teacher baselines need paired complete held-out losses")
    means=[float(x.mean()) for x in arrays]
    return {'passed_fit_internal_teacher_check':means[0]<means[1] and means[0]<means[2],
            'teacher_nll':means[0],'uniform_nll':means[1],'frequency_nll':means[2],
            'identity_quality_established':False}


def pack_game(results,decisions,*,target_color,game_plies,mode='raw_residual'):
    """41D per-game features with equal present-phase means; matched control dimensions.

    raw_residual includes the3 confidence descriptors; raw_confidence is an
    additional engineering diagnostic for attributing a gain to directionality.
    No control budget or main pilot has been frozen by this function.
    """
    if not results or len(results)!=len(decisions) or mode not in ('raw','confidence','raw_confidence','raw_residual') or target_color not in ('B','W') or game_plies<1:
        raise ContractError('P3 game aggregation contract mismatch')
    if any(d.target_color!=target_color or d.ply>game_plies for d in decisions) or len({d.ply for d in decisions})!=len(decisions):
        raise ContractError('P3 target/stage rows inconsistent or repeated')
    phases=[];counts=[]
    for stage in range(3):
        idx=[i for i,d in enumerate(decisions) if d.stage==stage]
        counts.append(len(idx))
        if not idx:continue
        rows=[]
        for i in idx:
            row=results[i]
            if np.shape(row['raw'])!=(16,) or np.shape(row['residual'])!=(16,):
                raise ContractError('Fixed semantic phi dimensionality changed')
            raw=row['raw'] if mode!='confidence' else np.zeros(16)
            residual=row['residual'] if mode=='raw_residual' else np.zeros(16)
            conf=np.asarray([row['log_probability']/math.log(362),row['entropy']/math.log(362),row['policy_rank']/362]) if mode!='raw' else np.zeros(3)
            rows.append(np.concatenate((raw,residual,conf)))
        phases.append(np.mean(rows,axis=0))
    output=np.concatenate((np.mean(phases,axis=0),[target_color=='B',target_color=='W',min(game_plies/720,1)],np.asarray(counts)/sum(counts))).astype(np.float32)
    if output.shape!=(41,) or not np.isfinite(output).all():raise ContractError('P3 game features invalid')
    return output
