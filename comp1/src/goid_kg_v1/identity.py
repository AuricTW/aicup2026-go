"""Train-only PLDA / SupCon head and paired, opened-development screening."""
from pathlib import Path
import time,random,json
import numpy as np
from . import native as n
from goid_v4 import score_push_v1 as c

def data():
    d=n.RUN/'feature_cache';receipt=n.read(d/'RECEIPT.json')
    assert n.sha(d/'vectors.npy')==receipt['vectors_sha256']
    indices=np.load(d/'indices.npy');valid=np.load(d/'valid.npy');x=np.load(d/'vectors.npy',mmap_mode='r')
    assert x.shape==(len(indices),2339) and valid.mean()>=.99 and np.isfinite(x).all()
    pos={int(index):j for j,index in enumerate(indices)}
    ti=np.load(n.RUN/'train_indices.npy');labels=np.load(n.RUN/'train_labels.npy')
    train=np.asarray([pos[int(i)] for i in ti]);ep=n.read(n.RUN/'PILOT_EPISODES.json')
    supports=[[pos[int(i)] for i in rows] for rows in ep['supports']]
    queries=[[pos[int(i)] for i in q['indices']] for q in ep['questions']]
    assert valid[train].mean()>=.99
    assert all(valid[rows].all() for rows in supports+queries),'Unsupported evaluation row: stop, do not replace zeros'
    train_ok=valid[train];train=train[train_ok];labels=labels[train_ok]
    return x,indices,train,labels,ep,supports,queries

def audit():
    x,indices,train,labels,ep,supports,queries=data();m=c.metadata()
    train_indices=indices[train];val_indices=indices[np.asarray(sorted({i for s in supports+queries for i in s}))]
    assert not set(m['owners'][train_indices])&set(m['owners'][val_indices])
    overlap={}
    for key in ('raw','move','canonical','group'):
        count=len(set(m[key][train_indices])&set(m[key][val_indices]));assert count==0,key;overlap[key]=count
    assert not set(i for s in supports for i in s)&set(i for s in queries for i in s)
    excluded=set(n.read(n.RUN/'EXPOSURE_CAPACITY.json')['future_target_players'])
    assert not set(m['owners'][train_indices].astype(str))&excluded
    n.write(n.RUN/'TRAIN_SCOPE_AUDIT.json',{'train_rows_valid':len(train),'train_players':len(np.unique(labels)),
        'candidate_players':len(ep['candidates']),'questions':len(queries),'unique_support_games':len({i for s in supports for i in s}),
        'unique_query_games':len({i for s in queries for i in s}),'owner_overlap':0,'support_query_row_overlap':0,
        'full_game_key_overlap':overlap,'features_finite':True,'source_original_safe_fit_only':True,
        'no_retired400_details_read':True,'no_independent_quality_claim':True})

def fit():
    import torch
    from threadpoolctl import threadpool_limits
    from goid_v4.plda_metric_v1 import fit as fit_plda
    from goid.feature_metric import FeatureMetric,PlayerBalancedSampler,supervised_contrastive_loss
    assert (n.RUN/'PREQUALITY_ADDENDUM.json').exists();audit()
    x,indices,train,labels,*_=data();d=n.RUN/'models';d.mkdir(exist_ok=False)
    t=time.perf_counter();trainraw=np.asarray(x[train],np.float64)
    mean=trainraw.mean(0);std=np.maximum(trainraw.std(0),1e-4)
    np.savez(d/'preprocess.npz',mean=mean,std=std)
    transformed=np.clip((np.asarray(x,np.float64)-mean)/std,-10,10).astype(np.float32)
    np.save(n.RUN/'standardized.npy',transformed,allow_pickle=False)
    receipts={}
    with threadpool_limits(limits=4):
        for name,columns,dimension in [('scalar_PLDA',slice(-35,None),16),('trunk_PLDA',slice(None),128)]:
            n.log(stage='fit_start',variant=name)
            receipts[name]=fit_plda(transformed[train,columns],labels,d/f'{name}.npz',dimension=dimension)
            n.log(stage='fit_done',variant=name,**receipts[name])
    device=n.runtime();torch.manual_seed(31415);torch.cuda.manual_seed_all(31415);random.seed(31415);np.random.seed(31415)
    model=FeatureMetric(2339,128,512,.2).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    sampler=PlayerBalancedSampler({int(i):int(y) for i,y in zip(train,labels)},16,4,31415)
    inp=transformed/np.maximum(np.linalg.norm(transformed,axis=1,keepdims=True),1e-12)
    tensor=torch.from_numpy(np.ascontiguousarray(inp)).to(device);history=[];start=time.perf_counter()
    fixedrows,fixedlabels=sampler.sample();model.eval()
    with torch.no_grad():before=float(supervised_contrastive_loss(model(tensor[fixedrows]),torch.from_numpy(fixedlabels).to(device),.1))
    model.train()
    for step in range(1,1001):
        rows,ys=sampler.sample();opt.zero_grad(set_to_none=True)
        z=model(tensor[rows]);loss=supervised_contrastive_loss(z,torch.from_numpy(ys).to(device),.1)
        assert torch.isfinite(loss);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5))
        assert np.isfinite(grad);opt.step()
        if step==1 or step%100==0:
            record={'step':step,'loss':float(loss.detach()),'gradient_norm':grad,'seconds':time.perf_counter()-start}
            history.append(record);n.log(stage='identity_head_train',**record)
    model.eval()
    with torch.no_grad():after=float(supervised_contrastive_loss(model(tensor[fixedrows]),torch.from_numpy(fixedlabels).to(device),.1))
    torch.save({'state_dict':model.cpu().state_dict(),'input_dim':2339,'hidden_dim':512,'embedding_dim':128,'dropout':.2,'seed':31415,'step':1000},d/'trunk_head.pt')
    receipts['trunk_head']={'steps':1000,'seed':31415,'seconds':time.perf_counter()-start,'loss_before_fixed_train_batch':before,
        'loss_after_fixed_train_batch':after,'history':history,'train_only_diagnostics_not_quality':True,'checkpoint':'last'}
    n.write(d/'TRAINING_RECEIPT.json',{'variants':receipts,'total_seconds':time.perf_counter()-t,
        'fit_train_rows':len(train),'fit_train_players':len(np.unique(labels)),
        'source_sha256':n.sha(Path(__file__)),'weights_sha256':{p.name:n.sha(p) for p in sorted(d.iterdir()) if p.is_file()}})

def predictions():
    import torch
    from threadpoolctl import threadpool_limits
    from goid_v4.plda_metric_v1 import PLDA,summarize
    from goid.feature_metric import FeatureMetric
    _,indices,_,_,ep,supports,queries=data()
    x=np.load(n.RUN/'standardized.npy',mmap_mode='r');d=n.RUN/'models';dest=n.RUN/'predictions';dest.mkdir(exist_ok=False)
    # This function never reads target/owner/rank fields to choose a score.
    with threadpool_limits(limits=4):
        for name,columns in [('scalar_PLDA',slice(-35,None)),('trunk_PLDA',slice(None))]:
            model=PLDA(d/f'{name}.npz');z=model.transform(x[:,columns]);qm,qc=summarize(z,queries);sm,sc=summarize(z,supports)
            score=model.score(qm,qc,sm,sc);np.save(dest/f'{name}.npy',score,allow_pickle=False)
    device=n.runtime();state=torch.load(d/'trunk_head.pt',map_location='cpu',weights_only=True)
    model=FeatureMetric(2339,128,512,.2).to(device);model.load_state_dict(state['state_dict']);model.eval()
    rows=[]
    with torch.inference_mode():
        for start in range(0,len(x),128):
            v=np.array(x[start:start+128],np.float32,copy=True);actual=len(v);v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
            v=np.pad(v,((0,128-actual),(0,0)))
            rows.append(model(torch.from_numpy(v).to(device)).cpu().numpy()[:actual])
    z=np.concatenate(rows).astype(np.float64)
    def prototypes(sets):
        a=np.stack([z[sorted(s)].mean(0) for s in sets]);return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
    qm,sm=prototypes(queries),prototypes(supports)
    score=np.stack([sm@np.ascontiguousarray(row) for row in qm]);np.save(dest/'trunk_head.npy',score,allow_pickle=False)
    ref=np.load(n.RUN/'F_projected100_reference.npy');assert ref.shape==score.shape
    for name in ('scalar_PLDA','trunk_PLDA','trunk_head'):
        np.save(dest/f'F_{name}.npy',c.fusion(ref,np.load(dest/f'{name}.npy')),allow_pickle=False)
    n.write(dest/'RECEIPT.json',{'questions':len(queries),'candidates':len(supports),'uses_labels_to_score':False,
        'no_assignment':True,'no_weight_sweep':True,'scores_sha256':{p.name:n.sha(p) for p in sorted(dest.glob('*.npy'))}})

def evaluate():
    ep=n.read(n.RUN/'PILOT_EPISODES.json');truth=ep['questions'];ref=np.load(n.RUN/'F_projected100_reference.npy')
    base,bv,br=c.metrics(ref,truth);dest=n.RUN/'evaluation';dest.mkdir(exist_ok=False)
    n.write(dest/'F_REFERENCE.json',base);results={};passing=[]
    players=sorted({q['player'] for q in truth});rowgroups=[[j for j,q in enumerate(truth) if q['player']==p and q['q'] in (5,10)] for p in players]
    for path in sorted((n.RUN/'predictions').glob('*.npy')):
        score=np.load(path);assert score.shape==ref.shape and np.isfinite(score).all()
        met,values,ranks=c.metrics(score,truth);delta={str(q):met[str(q)]['score']-base[str(q)]['score'] for q in (5,10,15,20)}
        short=met['short']-base['short'];perplayer=np.asarray([(values[ii]-bv[ii]).mean() for ii in rowgroups])
        rng=np.random.default_rng(20261104);boots=perplayer[rng.integers(0,len(players),size=(10000,len(players)))].mean(1)
        checks={'short_ge_003':short>=.003,'q5_ge0':delta['5']>=0,'q10_ge_minus005':delta['10']>=-.005,
                'q15_ge_minus015':delta['15']>=-.015,'q20_ge_minus015':delta['20']>=-.015}
        shortmask=np.asarray([q['q'] in (5,10) for q in truth]);rescued=int(((ranks==1)&(br!=1)&shortmask).sum());regressed=int(((ranks!=1)&(br==1)&shortmask).sum())
        result={'metrics':met,'delta_score':delta,'delta_short':short,'short_player_cluster_95_CI_descriptive':np.quantile(boots,[.025,.975]).tolist(),
                'short_top1_rescued':rescued,'short_top1_regressed':regressed,'checks':checks,'admitted':all(checks.values()),
                'exploratory_100_projected_not_independent_or_official':True}
        results[path.stem]=result;n.write(dest/f'{path.stem}.json',result)
        if result['admitted']:passing.append(path.stem)
        n.log(stage='pilot_evaluation',variant=path.stem,short_delta=short,q5_delta=delta['5'],q10_delta=delta['10'],admitted=result['admitted'])
    passing.sort(key=lambda name:(-results[name]['delta_short'],name))
    summary={'reference':base,'variants':results,'passing':passing,'winner':passing[0] if passing else None,
             'decision':'extend_one_frozen_winner' if passing else 'stop_no_HumanSL_submission_keep_F',
             'queries':400,'candidates':100,'unique_target_players':100,'all_opened_development':True,'no_official_or_independent_score':True,
             'retired400_evaluated':False,'official_test_forwarded':False,'uploaded':False}
    n.write(n.RUN/'PILOT_SUMMARY.json',summary)
    if not passing:n.write(n.RUN/'NO_NEW_CSV.json',{'reason':'All six predeclared variants failed screening; no official test run or HumanSL CSV generated',
        'retained_F_csv':str(n.ROOT/'submissions/20261104_F_sequence_v1/06_F_E_OpeningSequence.csv')})

def run():fit();predictions();evaluate()
