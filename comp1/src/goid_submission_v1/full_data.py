"""All OFFICIAL TRAINING data, explicitly authorized for a new competition refit.

No official query/candidates, old retired lists or quality matrices are read.
Historical self-test identities now join fitting; they are not validation for
these new weights. Last checkpoint and evaluated hyperparameters are fixed.
"""
from __future__ import annotations
from collections import defaultdict,Counter
from pathlib import Path
import argparse,hashlib,json,sqlite3,time
import numpy as np
import torch
from goid import inference as base,fixed_batch
from goid.feature_metric import FeatureMetric,PlayerBalancedSampler,supervised_contrastive_loss,FrozenFeaturePreprocessor
from goid_v4.pilot_models_v1 import MeanCount,fixed_encode
from goid_v4.pilot_train_v1 import runtime,numeric_probe
from goid_v4.p1 import compose_game_features,uncertainty_logits,identity_loss
from goid_v4.csv_u0_v1 import write

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/submission_20261101_v1_20261026'
FULL=RUN/'C_full_train';RANKS=('D','C','B','A','1D','2D','3D','4D','5D','6D')

def hist(raw,schema):
    x=np.asarray(raw[:,12:],np.float32).copy();offset=0
    for g in schema['groups']:
        if g['kind']!='histogram':continue
        width=g['stop']-g['start'];v=x[:,offset:offset+width];v/=np.maximum(v.sum(1,keepdims=True),np.float32(1e-12));offset+=width
    if offset!=1582 or not np.isfinite(x).all():raise ValueError('Histogram schema/value failure')
    return x

def prepare():
    start=time.perf_counter();FULL.mkdir(exist_ok=False);d=FULL/'data';d.mkdir()
    plan=base.read_json(RUN/'PLAN.json')
    if plan['status']!='fixed_before_full_data_fit':raise ValueError('No locked full-data protocol')
    proof=base.read_json(ROOT/'runs/research_v4/p1_u0_outer_v1_20261026/outer_reveal_v1/SUMMARY.json')
    if proof['G2']['status']!='passed' or proof['G2']['route']!='prospective_blocked_internal':raise ValueError('Evaluated U0 recipe evidence missing')
    audit=base.read_json(ROOT/'data/processed/audit_summary.json');integrity=base.read_json(ROOT/'data/processed/integrity_summary.json')
    if audit['parse_counts']!={'ok':1000000} or integrity['invalid_values']!=0:raise ValueError('Full source audit incomplete')
    source=ROOT/'data/processed/audit.sqlite';feature=ROOT/'data/processed/features.npy'
    before={str(p.relative_to(ROOT)):(p.stat().st_size,p.stat().st_mtime_ns) for p in (source,feature)}
    expected={Path(r['path']).name for r in base.read_json(ROOT/'data/raw/training_manifest.json')['files']}
    n=1000000;owners=np.empty(n,'S64');keys=np.empty(n,'S64');ranks=np.empty(n,np.uint8);colors=np.empty(n,np.uint8)
    moves=np.empty(n,np.uint16);target=np.empty(n,np.uint16);keep=np.empty(n,bool);seen=set();player_codes={};counts=Counter()
    parser=base.sha(ROOT/'src/goid/sgf_features.py');sgfversion='1.1.1'
    db=sqlite3.connect(source.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);db.execute('PRAGMA query_only=ON')
    sql='SELECT row_id,player_id,rank,color,mainline_moves,target_moves,COALESCE(split_group_hash,canonical_hash),raw_hash,source_file,parse_ok FROM games ORDER BY row_id'
    i=0
    try:
        for rid,p,rank,color,m,t,group,raw,filename,ok in db.execute(sql):
            if rid!=i or i>=n or ok!=1 or Path(filename).name not in expected or rank not in RANKS or color not in ('B','W') or not 0<t<=m<=65535:raise ValueError('Official training row identity/schema failure')
            code=player_codes.setdefault(p,hashlib.sha256(('goid-v4-player-v1|'+p).encode()).hexdigest())
            owners[i]=code;keys[i]=hashlib.sha256(f'goid-feature-cache-v1|{parser}|{sgfversion}|{raw}|{color}'.encode()).hexdigest()
            ranks[i]=RANKS.index(rank);colors[i]=int(color=='W');moves[i]=m;target[i]=t
            keep[i]=group not in seen;seen.add(group);counts[Path(filename).name]+=1;i+=1
    finally:db.close()
    if i!=n or len(seen)!=999990 or any(c!=100000 for c in counts.values()) or set(counts)!=expected:raise ValueError('Full data counts/dedup changed')
    del seen,player_codes
    fit_rows=np.flatnonzero(keep);order=np.argsort(keys[fit_rows],kind='stable');feature_rows=fit_rows[order]
    if len(set(keys.tolist()))!=n:raise ValueError('Repeated content-key')
    np.save(d/'fit_rows.npy',fit_rows,allow_pickle=False);np.save(d/'feature_rows.npy',feature_rows,allow_pickle=False)
    np.savez(d/'metadata.npz',owners=owners,keys=keys,ranks=ranks,colors=colors,moves=moves,target=target)
    raw=np.load(feature,mmap_mode='r',allow_pickle=False)
    if raw.shape!=(n,1594) or raw.dtype!=np.float32:raise ValueError('Source features differ')
    after={str(p.relative_to(ROOT)):(p.stat().st_size,p.stat().st_mtime_ns) for p in (source,feature)}
    if before!=after:raise ValueError('Original source mutated')
    write(FULL/'SCOPE.json',{'status':'all_official_training_authorized','source_rows':n,'effective_rows':len(fit_rows),'players':len(set(owners.tolist())),
        'retired_membership_lists_opened':False,'historical_self_tests_included_as_train_not_validation':True,'official_test_used_for_fit':False,
        'dedup':'first row_id per conservative effective group; global10 rows removed',
        'source_metadata_query':'read-only official training metadata only; full training scope newly authorized; no SGF column',
        'source_files':dict(counts),'source_size_mtime_before':before,'source_size_mtime_after':after,
        'source_feature_sha256':base.sha(feature),'metadata_sha256':base.sha(d/'metadata.npz'),'fit_rows_sha256':base.sha(d/'fit_rows.npy'),
        'feature_rows_sha256':base.sha(d/'feature_rows.npy'),'plan_sha256':base.sha(RUN/'PLAN.json'),'source_code_sha256':base.sha(Path(__file__)),
        'prior_B_validation_only_for_method':proof['G2'],'elapsed_seconds':time.perf_counter()-start})
    print(json.dumps({'stage':'scope','rows':len(fit_rows),'players':len(set(owners.tolist())),'seconds':time.perf_counter()-start}),flush=True)

def metadata():
    d=FULL/'data';scope=base.read_json(FULL/'SCOPE.json')
    if base.sha(d/'metadata.npz')!=scope['metadata_sha256']:raise ValueError('Metadata changed')
    z=np.load(d/'metadata.npz',allow_pickle=False);return {k:z[k] for k in z.files}

def train_base():
    start=time.perf_counter();d=FULL/'data';b=FULL/'base_bundle';b.mkdir(exist_ok=False);meta=metadata()
    rows=np.load(d/'fit_rows.npy',allow_pickle=False);raw=np.load(ROOT/'data/processed/features.npy',mmap_mode='r',allow_pickle=False)
    schema=base.read_json(ROOT/'artifacts/inference_bundle_v4_u0_v1/refit/feature_schema.json');write(b/'feature_schema.json',schema)
    total=np.zeros(1582,np.float64);sq=np.zeros_like(total)
    for i in range(0,len(rows),4096):
        v=hist(raw[rows[i:i+4096]],schema).astype(np.float64);total+=v.sum(0);sq+=(v*v).sum(0)
    mean=total/len(rows);std=np.maximum(np.sqrt(np.maximum(sq/len(rows)-mean*mean,0)),.001)
    np.savez(b/'fit_transform.npz',selected_indices=np.arange(12,1594),mean=mean.astype(np.float32),std=std.astype(np.float32))
    pre=FrozenFeaturePreprocessor(b,'standardized',10)
    normalized=np.lib.format.open_memmap(d/'normalized.npy',mode='w+',dtype=np.float32,shape=(len(rows),1582))
    for i in range(0,len(rows),2048):normalized[i:i+2048]=pre(raw[rows[i:i+2048]])
    normalized.flush();device=runtime();torch.manual_seed(20261025)
    x=torch.from_numpy(normalized).to(device);owners={i:meta['owners'][k].decode() for i,k in enumerate(rows)}
    sampler=PlayerBalancedSampler(owners,16,4,20261025);model=FeatureMetric(1582,128,512,.1).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);t=time.perf_counter();step=0;torch.cuda.reset_peak_memory_stats()
    write(b/'PREREG.json',{'steps':5000,'seed':20261025,'eligible_metric_players':len(sampler.eligible),'fit_rows':len(rows),'all_players':len(sampler.all_players),
        'checkpoint':'last','quality_selection':False,'code_sha256':base.sha(Path(__file__)),'scope_sha256':base.sha(FULL/'SCOPE.json')})
    try:
        with (b/'training.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,5001):
                if time.perf_counter()-t>3600:raise RuntimeError('MLP wall cap')
                ids,labels=sampler.sample();z=model(x[torch.from_numpy(ids).to(device)])
                loss=supervised_contrastive_loss(z,torch.from_numpy(labels).to(device),.1)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
                if torch.cuda.max_memory_allocated()>11*2**30:raise RuntimeError('MLP memory cap')
                if step==1 or step%1000==0:
                    r={'step':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-t,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30};log.write(json.dumps(r)+'\n');log.flush();print(json.dumps({'stage':'MLP',**r}),flush=True)
        with (b/'model_state.pt').open('xb') as f:torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},f)
        write(b/'RESULT.json',{'status':'complete_all_train_not_validation','steps':5000,'train_seconds':time.perf_counter()-t,'total_seconds':time.perf_counter()-start,
            'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'model_sha256':base.sha(b/'model_state.pt'),'old_weights_loaded':False})
    except BaseException as e:
        write(b/'FAILURE.json',{'step':step,'error':str(e),'elapsed_seconds':time.perf_counter()-t})
        with (b/'failed_model.pt').open('xb') as f:torch.save(model.state_dict(),f)
        raise

def encode():
    start=time.perf_counter();d=FULL/'data';b=FULL/'base_bundle';meta=metadata();rows=np.load(d/'feature_rows.npy',allow_pickle=False)
    raw=np.load(ROOT/'data/processed/features.npy',mmap_mode='r',allow_pickle=False)
    result=base.read_json(b/'RESULT.json')
    if result['steps']!=5000 or base.sha(b/'model_state.pt')!=result['model_sha256']:raise ValueError('Base not fixed')
    bundle={'path':b,'manifest':{'synthetic_control':False,'model_config':{'embedding_dim':128,'hidden_dim':512,'dropout':.1}}}
    model,pre=fixed_batch.frozen_model(bundle,'cuda');before=fixed_batch.model_digest(model)
    features=np.lib.format.open_memmap(d/'features.npy',mode='w+',dtype=np.float32,shape=(len(rows),1728))
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        for i in range(0,len(rows),1024):
            ids=rows[i:i+1024];v=np.array(raw[ids],copy=True);stat=base.statistical_vectors(v,bundle)
            mlp=model(torch.from_numpy(fixed_batch.padded_input(pre(v))).to('cuda')).cpu().numpy()[:len(ids)]
            stages=np.rint(v[:,9:12].astype(np.float64)*meta['target'][ids,None])
            if not np.array_equal(stages.sum(1),meta['target'][ids]):raise ValueError('Stage reconstruction failed')
            features[i:i+len(ids)]=compose_game_features(stat,mlp,v[:,:12],['B' if c==0 else 'W' for c in meta['colors'][ids]],meta['moves'][ids],stages)
    features.flush()
    if before!=fixed_batch.model_digest(model):raise ValueError('Encoder changed')
    write(d/'ENCODING.json',{'status':'complete','games':len(rows),'dimension':1728,'fixed_game_batch':1024,'features_sha256':base.sha(d/'features.npy'),
        'model_before':before,'model_after':fixed_batch.model_digest(model),'elapsed_seconds':time.perf_counter()-start})
    print(json.dumps({'stage':'encoded','games':len(rows),'seconds':time.perf_counter()-start}),flush=True)

class IndexedSampler:
    """Same episodic sampling as pilot TensorSampler, using full-array indices."""
    def __init__(self,by_player,rank,seed):
        self.by_player={p:np.asarray(ids,np.int64) for p,ids in by_player.items() if len(ids)>=120};self.rank=rank;self.rng=np.random.default_rng(seed)
        self.by_rank=defaultdict(list)
        for p in self.by_player:self.by_rank[rank[p]].append(p)
        self.ranks=sorted(self.by_rank)
        for ps in self.by_rank.values():ps.sort()
        if len(self.ranks)!=10 or any(len(v)<16 for v in self.by_rank.values()):raise ValueError('Rank batch capacity')
    def sample(self):
        selected=self.rng.choice(self.ranks,2,replace=False);players=[p for rank in selected for p in self.rng.choice(self.by_rank[rank],16,replace=False)]
        ids=np.zeros((64,100),np.int64);mask=np.zeros_like(ids,bool)
        for j,p in enumerate(players):
            q=int(self.rng.choice([5,10,20],p=[.4,.4,.2]));chosen=self.rng.choice(self.by_player[p],100+q,replace=False)
            ids[j]=sorted(chosen[:100]);mask[j]=True;ids[j+32,:q]=sorted(chosen[100:]);mask[j+32,:q]=True
        negative=np.ones((32,32),bool)
        for i,p in enumerate(players):
            cross=[j for j,o in enumerate(players) if self.rank[o]!=self.rank[p]];negative[i,int(self.rng.choice(cross))]=False
        return ids,mask,negative

def head():
    start=time.perf_counter();d=FULL/'data';out=FULL/'head';out.mkdir(exist_ok=False);meta=metadata();rows=np.load(d/'feature_rows.npy',allow_pickle=False)
    encoded=base.read_json(d/'ENCODING.json')
    if base.sha(d/'features.npy')!=encoded['features_sha256']:raise ValueError('Features changed')
    features=np.load(d/'features.npy',mmap_mode='r',allow_pickle=False);by_player=defaultdict(list);rank={}
    for i,k in enumerate(rows):
        p=meta['owners'][k].decode();by_player[p].append(i);rank[p]=RANKS[int(meta['ranks'][k])]
    sampler=IndexedSampler(by_player,rank,31415)
    prior=float(np.mean([np.var(np.asarray(features[ids,1582:1710],np.float64),axis=0,ddof=1).mean() for ids in sampler.by_player.values()]))
    floor=prior/100*.01;device=runtime();torch.manual_seed(31415);np.random.seed(31415)
    x=torch.from_numpy(features).to(device);colors=torch.tensor(meta['colors'][rows].astype(np.int64),device=device)
    model=MeanCount(floor=floor,prior=prior).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    # Same100 former calibration-source identities, now train. Preserve their
    #20 query-source games where retained; deterministic within-player fallback.
    old=base.read_json(ROOT/'runs/research_v4/p1_pilot_v1_20261026/data/cal_records.json');keypos={meta['keys'][k].decode():i for i,k in enumerate(rows)}
    oldgroups=defaultdict(list)
    for r in old:oldgroups[r['player']].append(r['key'])
    singles=[];replaced=0
    for p,kk in sorted(oldgroups.items()):
        preferred=sorted(kk)[100:120];positions=[keypos[k] for k in preferred if k in keypos]
        if len(positions)<20:
            alternatives=[i for i in by_player[p] if i not in positions];positions+=alternatives[:20-len(positions)];replaced+=1
        singles.extend([[i] for i in positions])
    if len(oldgroups)!=100 or len(singles)!=2000:raise ValueError('Prior-source scope mismatch')
    ps=sorted(sampler.by_player)[:100];supports=[sampler.by_player[p][:100].tolist() for p in ps]
    queries=[sampler.by_player[p][100:100+q].tolist() for p in ps[:10] for q in (5,10,20)]
    write(out/'PREREG.json',{'steps':5000,'seed':31415,'checkpoint':'last','initial_prior':prior,'initial_floor':floor,
        'head_players':len(sampler.by_player),'head_eligible_games':sum(map(len,sampler.by_player.values())),'head_players_per_rank':{k:len(v) for k,v in sampler.by_rank.items()},
        'no_heldout_quality_or_grid_selection':True,'final_floor_multiplier':.001,'variance_scale':1.,'fusion_new_rank_weight':.25,
        'former_calibration_players_now_training':100,'prior_replacement_players_due_dedup':replaced,'code_sha256':base.sha(Path(__file__))})
    model.eval();write(out/'NUMERIC_BEFORE.json',numeric_probe(model,x,colors,supports,queries));model.train();t=time.perf_counter();step=0;torch.cuda.reset_peak_memory_stats()
    try:
        with (out/'training.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,5001):
                if time.perf_counter()-t>3600:raise RuntimeError('Head wall cap')
                ids,mask,negative=sampler.sample();ti=torch.from_numpy(ids).to(device);tm=torch.from_numpy(mask).to(device)
                optimizer.zero_grad(set_to_none=True);mean,var=model.forward_packed(x[ti].masked_fill(~tm[:,:,None],0),colors[ti],tm)
                loss=identity_loss(uncertainty_logits(mean[32:],var[32:],mean[:32],var[:32]),negative_mask=torch.from_numpy(negative).to(device))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
                if torch.cuda.max_memory_allocated()>11*2**30:raise RuntimeError('Head memory cap')
                if step==1 or step%500==0:
                    r={'step':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-t,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30};log.write(json.dumps(r)+'\n');log.flush();print(json.dumps({'stage':'head',**r}),flush=True)
        training_seconds=time.perf_counter()-t;model.eval().requires_grad_(False)
        mean,_=fixed_encode(model,x,colors,singles);final_prior=max(float(mean.reshape(100,20,128).astype(np.float64).var(1,ddof=1).mean()),1e-8)
        with torch.no_grad():model.prior_variance.fill_(final_prior)
        model.floor=final_prior/100*.001;write(out/'NUMERIC_FINAL.json',numeric_probe(model,x,colors,supports,queries))
        with (out/'model.pt').open('xb') as f:torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'method':'U0_mean_count',
            'seed':31415,'steps':5000,'floor':model.floor,'variance_scale':1.},f)
        write(out/'RESULT.json',{'status':'complete_competition_all_train_no_independent_quality','steps':5000,'training_seconds':training_seconds,'total_seconds':time.perf_counter()-start,
            'final_prior':final_prior,'floor':model.floor,'floor_multiplier':.001,'variance_scale':1.,'alpha':.25,
            'model_sha256':base.sha(out/'model.pt'),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
            'no_validation_accuracy_computed':True,'old_self_tests_are_now_training':True})
    except BaseException as e:
        write(out/'FAILURE.json',{'step':step,'error':str(e),'elapsed_seconds':time.perf_counter()-t})
        with (out/'failed_model.pt').open('xb') as f:torch.save(model.state_dict(),f)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train_base','encode','head']);a=p.parse_args();globals()[a.stage]()
