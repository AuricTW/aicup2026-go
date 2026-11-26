"""Read-only oldcal fusion and player-paired inference comparison utilities."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];PARENT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from go_rank.submission_prep.shared import pool
from go_rank.katago_backbone_v1.evaluate import mix,metrics
LABELS=('D','C','B','A','1D','2D','3D','4D','5D','6D');SEEDS=(42,123,2026)
def queries():
    qs=[]
    for fold in ('A','B'):
        for line in (ROOT/f'runs/katago_backbone_20261104_v1/{fold}/cal_queries.jsonl').read_text(encoding='utf-8-sig').splitlines():q=json.loads(line);q['_fold']=fold;qs.append(q)
    assert len(qs)==1600 and len({q['player_id'] for q in qs})==400;return qs
def reward(y,p):d=abs(y-p.argmax(1));return (d==0).astype(float)+(d==1)*np.exp(-1)
def read_cnn(qs,base_paths,q_paths):
    short=np.array([q['num_games']==5 for q in qs]);bases=[];qps=[];rows=None
    for basepath,qpath in zip(base_paths,q_paths):
        b=np.load(basepath);q=np.load(qpath);assert np.array_equal(b['rows'],q['rows']);rows=b['rows'];bases.append(b['probabilities']);qps.append(q['probabilities'])
    singles=np.array([np.where(short[:,None],pool(b,rows,qs),pool(q,rows,qs)) for b,q in zip(bases,qps)]);ensemble=np.where(short[:,None],pool(np.mean(bases,0),rows,qs),np.mean([pool(q,rows,qs) for q in qps],0));return ensemble,singles
def fullsystem(qs,cnn,seed_cnn):
    sp=[]
    for seed in SEEDS:
        p=np.load(PARENT/f'fresh_sp/seed{seed}/cal_posteriors.npz');assert np.array_equal(p['question_ids'],[q['question_id'] for q in qs]);sp.append(p['probabilities'])
    ce=np.load(PARENT/'old20_fixed_CAL_FUSED.npz')['pure_CE_probabilities'];v6=mix(cnn,np.mean(sp,0),.25);full=mix(v6,ce.mean(0),.25);single=np.array([mix(mix(a,b,.25),c,.25) for a,b,c in zip(seed_cnn,sp,ce)]);return full,single,v6
def compare(qs,reference,candidate,refseed,candseed,family):
    y=np.array([q['rank_idx'] for q in qs]);delta=reward(y,candidate)-reward(y,reference);players={}
    for q,d in zip(qs,delta):players.setdefault((q['_fold'],q['player_id'],int(q['rank_idx'])),[]).append((q['num_games'],float(d)))
    assert len(players)==400 and all(len(v)==4 for v in players.values());strata=[];folds={}
    for fold in ('A','B'):
        ix=np.array([q['_fold']==fold for q in qs]);folds[fold]=dict(delta=float(delta[ix].mean()),seed_deltas=[float((reward(y[ix],b[ix])-reward(y[ix],a[ix])).mean()) for a,b in zip(refseed,candseed)])
        for rank in range(10):v=np.array([np.mean([d for _,d in part]) for (ff,p,r),part in players.items() if ff==fold and r==rank]);assert len(v)==20;strata.append(v)
    rng=np.random.default_rng(20261106);boot=np.zeros(20000)
    for v in strata:
        for lo in range(0,20000,500):boot[lo:lo+500]+=rng.choice(v,(500,20),replace=True).sum(1)/400
    ci=np.quantile(boot,[.05/(2*family),1-.05/(2*family)]).tolist();byG={str(g):float(delta[np.array([q['num_games']==g for q in qs])].mean()) for g in (5,10,15,20)};byR={LABELS[r]:float(delta[y==r].mean()) for r in range(10)};mean=float(delta.mean());gates=dict(delta=mean>=.01,CI_lower_positive=ci[0]>0,bothfoldspositive=all(v['delta']>0 for v in folds.values()),twoofthreeseeds_eachfold=all(sum(d>0 for d in v['seed_deltas'])>=2 for v in folds.values()),allG=min(byG.values())>=-.005,protected_ranks=all(byR[r]>=-.01 for r in ('1D','2D','4D')),allclasses=min(byR.values())>=-.02);return dict(delta=mean,CI=ci,family_comparisons=family,players=400,queries=1600,folds=folds,byG_delta=byG,byRank_delta=byR,gates=gates,go=all(gates.values()),scope='Conditionalmodel/selection, repeatedlyusedoldcal;notnewconfirmation/Public')
