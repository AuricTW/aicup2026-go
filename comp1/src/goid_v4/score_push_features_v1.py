"""Train-background-smoothed sparse sequence models, no test adaptation."""
from __future__ import annotations
import sqlite3,time
from pathlib import Path
import numpy as np
from scipy import sparse
from . import score_push_v1 as c

BINS=65536
VARIANTS=('bag_position','target_order','joint_order_absolute','joint_order_D4')

def sequence_export(indices, m):
    """Explicit safe-row primary-key lookup; no owners/truth returned by DB."""
    indices=np.asarray(indices,np.int64);rows=m['rows'][indices]
    db=sqlite3.connect((c.ROOT/'data/processed/audit.sqlite').resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    arr=np.full((len(rows),40),65535,np.uint16);pos={int(r):i for i,r in enumerate(rows)}
    try:
        db.execute('PRAGMA query_only=ON')
        for start in range(0,len(rows),800):
            batch=[int(x) for x in rows[start:start+800]]
            sql=f'SELECT row_id,move_sequence FROM games WHERE row_id IN ({",".join("?" for _ in batch)})'
            plans=[r[3] for r in db.execute('EXPLAIN QUERY PLAN '+sql,batch)]
            assert all('SCAN' not in p for p in plans) and any('INTEGER PRIMARY KEY' in p for p in plans)
            fetched=db.execute(sql,batch).fetchall();assert {r[0] for r in fetched}==set(batch)
            for row,blob in fetched:
                z=np.frombuffer(blob,dtype='<u2')[:40]
                # Convert cached pass32767 to361 and white bit to362 offset.
                codes=(z&32767).astype(np.uint16);codes[codes==32767]=361
                codes+=((z>>15)*362).astype(np.uint16)
                arr[pos[row],:len(codes)]=codes
    finally:db.close()
    return arr

def d4_table():
    from .replay import transform_action
    table=np.asarray([[transform_action(p,19,s) for p in range(362)] for s in range(8)],np.uint64)
    return np.concatenate((table,table+362,np.full((8,1),724,np.uint64)),axis=1)

def hashed(code):
    with np.errstate(over='ignore'):
        x=np.asarray(code,np.uint64)+np.uint64(0x9e3779b97f4a7c15)
        x=(x^(x>>30))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>27))*np.uint64(0x94d049bb133111eb)
        return ((x^(x>>31))&np.uint64(BINS-1)).astype(np.int32)

def design(arr,variant,*,block=2048):
    if variant not in VARIANTS:raise ValueError(variant)
    rr=[];cc=[];vv=[];table=d4_table() if variant=='joint_order_D4' else None
    groups=1 if variant=='bag_position' else 3
    for start in range(0,len(arr),block):
        codes=arr[start:start+block].copy();valid=codes!=65535
        if variant=='target_order':
            target=(codes<362)&valid
            # All targets are B, compress preserving chronological order.
            new=np.full((len(codes),40),65535,np.uint16)
            ir,ic=np.nonzero(target);new[ir,np.cumsum(target,axis=1)[ir,ic]-1]=codes[ir,ic]
            codes=new;valid=codes!=65535
        safe=np.where(valid,codes,724).astype(np.uint64)
        for group,n in enumerate((1,) if groups==1 else (2,3,4)):
            mask=valid[:,:41-n].copy()
            for k in range(1,n):mask &=valid[:,k:41-n+k]
            if variant!='target_order':mask &=codes[:,n-1:]<362
            if table is not None:
                transformed=table[:,safe]
                val=np.zeros((8,len(codes),41-n),np.uint64)
                for k in range(n):val=val*725+transformed[:,:,k:41-n+k]
                val=val.min(axis=0)
            else:
                val=np.zeros((len(codes),41-n),np.uint64)
                for k in range(n):val=val*725+safe[:,k:41-n+k]
            # Stage is retained for ordered variants; bag deliberately unordered.
            if groups>1:val=val*4+(np.arange(n-1,40)//10)[None,:]
            rows,cols=np.nonzero(mask);denom=mask.sum(axis=1)
            rr.append((rows+start).astype(np.int32));cc.append(hashed(val[rows,cols])+group*BINS)
            vv.append((1/np.maximum(denom[rows],1)).astype(np.float32))
    x=sparse.csr_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(len(arr),groups*BINS),dtype=np.float32)
    x.sum_duplicates();x.sort_indices()
    return x

def likelihood(x,fit_positions,support_positions,query_positions,*,alpha=20.):
    """Exact posterior-predictive token LLR vs train background, support-only sufficient stats.

    One game contributes mass1 per available ngram group, not per move. The
    omitted background log likelihood is query-constant across candidates.
    """
    groups=x.shape[1]//BINS
    bg=np.asarray(x[fit_positions].sum(axis=0)).ravel().astype(np.float64)+1e-3
    for g in range(groups):bg[g*BINS:(g+1)*BINS]/=bg[g*BINS:(g+1)*BINS].sum()
    bank=sparse.vstack([sparse.csr_matrix(x[s].sum(axis=0)) for s in support_positions],format='csr',dtype=np.float64)
    norm=np.stack([np.asarray(bank[:,g*BINS:(g+1)*BINS].sum(1)).ravel() for g in range(groups)],axis=1)
    bank.data=np.log1p(bank.data/(alpha*bg[bank.indices]))
    used=sorted({i for qs in query_positions for i in qs});qx=x[used]
    valid=np.stack([np.asarray(qx[:,g*BINS:(g+1)*BINS].sum(1)).ravel() for g in range(groups)],axis=1)
    scores=(qx@bank.T).toarray()-valid@np.log1p(norm/alpha).T
    positions={j:i for i,j in enumerate(used)}
    out=np.stack([scores[[positions[j] for j in qs]].mean(axis=0)/groups for qs in query_positions])
    assert np.isfinite(out).all()
    return out,{'groups':groups,'nnz':x.nnz,'rows':x.shape[0],'empty_games':int((x.getnnz(1)==0).sum()),
        'background_fit_rows':len(fit_positions),'alpha':alpha,'hash_bins_per_group':BINS,'background_comes_only_from_training':True}

def run():
    start=time.perf_counter();m=c.metadata();ep=c.read(c.RUN/'EPISODES.json')
    fit=np.load(c.RUN/'fitB_indices.npy',allow_pickle=False)
    val=sorted({i for s in ep['supports'] for i in s}|{i for q in ep['questions'] for i in q['indices']})
    used=np.asarray(sorted(set(fit.tolist())|set(val)),np.int64);positions={v:i for i,v in enumerate(used)}
    assert not set(fit.tolist())&set(val)
    dest=c.RUN/'F';dest.mkdir(exist_ok=False)
    c.write(dest/'HASHING_ADDENDUM.json',{'before_any_F_quality':True,'fixed_hash_space':BINS,
        'hash_space_has_no_test_fitted_vocabulary':True,'train_only_background':True,'no_background_estimated_from_query':True,
        'sequence_source':'already audited move_sequence BLOB by exact safe primary keys;HD verified by scope metadata',
        'variants':list(VARIANTS),'code_sha256':c.sha(Path(__file__))})
    arr=sequence_export(used,m);np.save(dest/'opening_codes.npy',arr,allow_pickle=False);np.save(dest/'indices.npy',used,allow_pickle=False)
    fp=[positions[i] for i in fit];sp=[[positions[i] for i in s] for s in ep['supports']]
    qp=[[positions[i] for i in q['indices']] for q in ep['questions']]
    reference=np.load(c.RUN/'selection_prediction/RRF_D_PLDA_raw.npy')
    from threadpoolctl import threadpool_limits
    for variant in VARIANTS:
        t=time.perf_counter()
        with threadpool_limits(limits=2):
            x=design(arr,variant);score,receipt=likelihood(x,fp,sp,qp)
        np.save(dest/f'{variant}.npy',score,allow_pickle=False)
        receipt.update({'seconds':time.perf_counter()-t,'no_reserved323_rows_replayed':True,'source_sha256':c.sha(dest/'opening_codes.npy')})
        c.write(dest/f'{variant}_RECEIPT.json',receipt)
        c.evaluate_selection('F_'+variant,score)
        c.evaluate_selection('E_F_'+variant,c.fusion(reference,score))
        del x,score
        if time.perf_counter()-start>10800:raise TimeoutError('F wall budget')
    c.write(dest/'COMPLETE.json',{'seconds':time.perf_counter()-start,'code_sha256':c.sha(Path(__file__)),'reserved323_quality_read':False})

if __name__=='__main__':run()
