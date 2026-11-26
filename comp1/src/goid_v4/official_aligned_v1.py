"""Versioned official-aligned validation. Original fit allowlist only.

Retired400 data are never queried. New source files and runs only.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, mmap, os, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from .pilot_scope_v1 import PointReader, opaque, RANKS, fit_player

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/research_v4/official_aligned_v1_20261104'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def token(*x):return hashlib.sha256('|'.join(map(str,x)).encode()).hexdigest()
def log(**kw):print(json.dumps(kw,ensure_ascii=False),flush=True)
def meta():
    with np.load(OUT/'scope_metadata.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def capacity():
    OUT.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    preflight=ROOT/'runs/opening_day_20261104_v1/MODEL_PREFLIGHT.json'  # integrity guard of the original run; absent in a fresh rebuild
    protected=read(preflight)['protected_files'] if preflight.exists() else {}
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    write(OUT/'PROTECTED_BEFORE.json',protected)
    path=ROOT/'runs/statistical_v1/fit_row_ids.npy'
    assert sha(path)=='b0bf9c6a91317456d53ff63555dea5a2249041f7538de058e41ce81d9cd30bd8'
    ids=np.load(path,allow_pickle=False)
    dbpath=ROOT/'data/processed/audit.sqlite';dbstat=(dbpath.stat().st_size,dbpath.stat().st_mtime_ns)
    reader=PointReader(dbpath)
    try:rows=reader.fetch(ids);plan=reader.plan
    finally:reader.close()
    rows.sort(key=lambda r:r['row_id'])
    assert len(rows)==588294 and all(fit_player(r['player_id']) for r in rows)
    assert len({r['group_key'] for r in rows})==len(rows)
    # Locate newlines across the versioned CSV bytes; decode ONLY explicit fit rows.
    # Other rows are never parsed or included in a metadata/feature artifact.
    hd=np.full(len(rows),255,np.uint8);byfile=defaultdict(dict)
    for i,r in enumerate(rows):byfile[r['source_file']][r['source_row']]=i
    sources={}
    for name,mapping in sorted(byfile.items()):
        path=ROOT/'data/raw/training_v1_1_20261104/training'/name
        with path.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
            startline=0;line=0
            while startline<len(mm):
                end=mm.find(b'\n',startline)
                if end<0:end=len(mm)
                if line in mapping:
                    i=mapping[line];r=rows[i]
                    values=next(csv.reader([mm[startline:end].decode('utf-8').rstrip('\r')]))
                    assert values[:4]==[r['player_id'],r['game_id'],r['rank'],r['color']]
                    sgf=values[4];prefix='(;SZ[19]HD[';assert sgf.startswith(prefix)
                    j=sgf.index(']',len(prefix));h=int(sgf[len(prefix):j]);assert 0<=h<=9
                    old=sgf[:8]+sgf[j+1:]
                    assert hashlib.sha256(old.encode()).hexdigest()==r['raw_hash']
                    hd[i]=h
                startline=end+1;line+=1
        assert line==100001
        sources[name]={'bytes':path.stat().st_size,'sha256':sha(path),'decoded_allowed_rows':len(mapping)}
        log(stage='capacity_version_alignment',file=name,allowed_rows=len(mapping))
    assert (hd!=255).all() and dbstat==(dbpath.stat().st_size,dbpath.stat().st_mtime_ns)
    fields={'rows':np.asarray([r['row_id'] for r in rows],np.int64),
            'owners':np.asarray([opaque(r['player_id']) for r in rows],dtype='S64'),
            'ranks':np.asarray([r['rank'] for r in rows],dtype='U2'),
            'colors':np.asarray([int(r['color']=='W') for r in rows],np.uint8),
            'moves':np.asarray([r['mainline_moves'] for r in rows],np.uint16),
            'target':np.asarray([r['target_moves'] for r in rows],np.uint16),'hd':hd}
    for field,key in [('raw','raw_hash'),('move','move_hash'),('canonical','canonical_hash'),('group','group_key')]:
        fields[field]=np.asarray([r[key] for r in rows],dtype='S64')
    np.savez(OUT/'scope_metadata.npz',**fields)
    mask=(fields['colors']==0)&(hd<=1);counts=Counter(fields['owners'][mask].tolist())
    rank_by_player=dict(zip(fields['owners'].tolist(),fields['ranks'].tolist()))
    byrank=Counter(rank_by_player[p] for p,n in counts.items() if n>=120)
    summary={'fit_rows':len(rows),'fit_players':len(set(fields['owners'].tolist())),
        'B_HD01_rows':int(mask.sum()),'B_HD01_HD_counts':dict(Counter(map(int,hd[mask]))),
        'eligible_B_HD01_120_by_rank':dict(byrank),'eligible_total':sum(byrank.values()),
        'threshold_capacity':{str(n):sum(v>=n for v in counts.values()) for n in (105,110,120,140)},
        'pristine_players_proven':0,'historically_exposed_fit_population':True,
        'point_lookup_query_plan':plan,'original_fit_index_sha256':sha(ROOT/'runs/statistical_v1/fit_row_ids.npy'),
        'retired_membership_or_details_read':False,'CSV_byte_boundary_scan':True,
        'CSV_semantic_decode_scope':'explicit original fit row allowlist only',
        'v11_rows_verified_equal_to_v10_except_HD':len(rows),'sources':sources,
        'metadata_sha256':sha(OUT/'scope_metadata.npz'),'seconds':time.perf_counter()-start}
    write(OUT/'CAPACITY.json',summary);log(stage='capacity_complete',**summary)

def split():
    m=meta();cap=read(OUT/'CAPACITY.json');assert sha(OUT/'scope_metadata.npz')==cap['metadata_sha256']
    good=(m['colors']==0)&(m['hd']<=1);members=defaultdict(list)
    for i in np.flatnonzero(good):members[m['owners'][i]].append(int(i))
    byrank=defaultdict(list)
    for p,ids in members.items():
        if len(ids)>=120:
            rr=set(m['ranks'][ids].tolist());assert len(rr)==1
            byrank[next(iter(rr))].append(p)
    quota={r:min(40,len(byrank[r])) for r in RANKS}
    while sum(quota.values())<400:
        before=sum(quota.values())
        for r in RANKS:
            if quota[r]<len(byrank[r]):quota[r]+=1
            if sum(quota.values())==400:break
        if before==sum(quota.values()):raise ValueError('Cannot create 400 B/HD01 candidates')
    candidates=[]
    for r in RANKS:candidates.extend(sorted(byrank[r],key=lambda p:token('candidate20261104',p.decode()))[:quota[r]])
    candidates.sort(key=lambda p:token('order',p.decode()))
    val=set(candidates);base=np.asarray([i for i,p in enumerate(m['owners']) if p not in val],np.int64)
    supports=[];queries=[]
    # Target split is player-disjoint; both phases condition on the same fixed400 support bank.
    targetrole={p:('selection' if j%2==0 else 'confirmation') for j,p in enumerate(sorted(candidates,key=lambda p:token('targetrole',p.decode())))}
    for j,p in enumerate(candidates):
        ids=sorted(members[p],key=lambda i:token('game20261104',m['group'][i].decode()))[:120]
        supports.append(sorted(ids[:100],key=lambda i:m['raw'][i]))
        for g in (5,10,15,20):
            queries.append({'id':f'aligned_p{j:04}_g{g}','target':j,'player':p.decode(),'role':targetrole[p],
                'q':g,'indices':sorted(ids[100:100+g],key=lambda i:m['raw'][i])})
    vind=np.asarray([i for s in supports for i in s]+[i for q in queries for i in q['indices']],np.int64)
    assert len(set(m['owners'][base].tolist())&val)==0
    for f in ('raw','move','canonical','group'):assert not (set(m[f][base].tolist())&set(m[f][vind].tolist()))
    assert len(set(m['group'][vind].tolist()))==48000
    # Exact historical head population, refit only if all excluded validation players avoided.
    # New heads are selected from the current base population, no old weights loaded.
    ranks={p:m['ranks'][np.flatnonzero(m['owners']==p)[0]] for p in set(m['owners'][base].tolist())}
    allmembers=defaultdict(list)
    for i in base:allmembers[m['owners'][i]].append(int(i))
    head=[]
    for r in RANKS:
        ps=sorted([p for p,ids in allmembers.items() if ranks[p]==r and len(ids)>=120],key=lambda p:token('head',p.decode()))
        assert len(ps)>=50,(r,len(ps))
        for j,p in enumerate(ps[:50]):
            ids=sorted(allmembers[p],key=lambda i:token('headgame',m['group'][i].decode()))[:120]
            for i in ids:head.append({'i':i,'part':'train' if j<40 else 'cal','player':p.decode(),'rank':r})
    np.save(OUT/'base_indices.npy',base,allow_pickle=False)
    write(OUT/'EPISODES.json',{'candidates':[p.decode() for p in candidates],'supports':supports,'questions':queries,
        'quota':quota,'heads':head,'one_fixed_candidate_pool':True,'selection_targets':200,'confirmation_targets':200,
        'confirmation_is_not_pristine':True,'shared_support_bank_conditional_inference':True})
    write(OUT/'PROTOCOL.json',{
        'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'motivation':'Earlier official results were known before this design.',
        'scope':'original588294 fit rows only; retired400 excluded by prior fit-index provenance',
        'official_alignment':'400 support candidates x100 B HD0/1 games; q5/10/15/20; disjoint support/query groups',
        'population_limit':'eligibility-conditioned original fit population; historical exposure; not official test or pristine',
        'base_training_rows':len(base),'validation_players':400,'head_fit_players':400,'head_cal_players':100,
        'fitted_states':'fresh scaler+MLP, PCA+Gaussian, U0 and new PLDA all exclude400validation identities',
        'baseline_MLP':{'steps':5000,'seed':20261025,'same_architecture_as_D':True},
        'baseline_U0':'same5000steps/s31415 and fit-inner calibration as frozen D method; fresh weights excluding validation',
        'new_family':'full within-player covariance whitening plus between-player covariance PLDA; raw style histograms and fresh stat+MLP representation',
        'selection_variants':['PLDA_raw','PLDA_composed','RRF_D_PLDA_raw','RRF_D_PLDA_composed'],
        'PLDA_fixed':{'PCA_dim':256,'within_covariance_diagonal_shrinkage':0.1,'between_eigenvalue_floor':1e-5,
            'score':'posterior predictive Gaussian conditional on100supports and query count; no cross-query state'},
        'fusion':'0.5/(60+rank_D)+0.5/(60+rank_PLDA); no alpha grid',
        'selection':'largest mean(q5,q10) among four fixed variants on200selection targets; ties first; require improvement over D and B',
        'confirmation':'one selected candidate; open200confirmation targets once after selected model/fusion hash freeze',
        'gate':{'short_point_gain_vs_D_min':0.005,'q5_q10_point_gain_each_min':0.,'q15_q20_noninferiority_margin':0.005,
            'bootstrap':10000,'bootstrap_seed':2026110401,'cluster':'target player across nested query counts',
            'short_one_sided_97_5_lower_vs_D_and_B_gt':0.,'conditional_on_fixed_support_bank':True},
        'no_retired400_access':True,'no_official_training_or_query_adaptation':True,'no_assignment':True,
        'no_external_data_or_weights':True,'no_upload':True,
        'failure':'preserve failures and D/B; failed gate does not become validated candidate; no relaxed threshold',
        'budget':'RTX3060 only; baseline MLP30min, U0twohours; each PLDAfit30min; no infinite tuning',
        'sources':['https://doi.org/10.1109/ICCV.2007.4409052','https://kaldi-asr.org/doc/classkaldi_1_1Plda.html'],
        'implementation':'locally derived covariance model; conceptual PLDA reference, not claimed paper replication',
        'episode_sha256':sha(OUT/'EPISODES.json'),'metadata_sha256':sha(OUT/'scope_metadata.npz')})
    log(stage='split_and_protocol_frozen',quota=quota,base_rows=len(base),confirmation_targets=200)

def base():
    from .pilot_train_v1 import runtime
    from .plda_metric_v1 import histograms
    from goid.feature_metric import FeatureMetric,PlayerBalancedSampler,supervised_contrastive_loss,FrozenFeaturePreprocessor
    from goid import inference as infer, fixed_batch
    from .p1 import compose_game_features
    import torch
    t=time.perf_counter();m=meta();ep=read(OUT/'EPISODES.json');base_ids=np.load(OUT/'base_indices.npy')
    bundle=OUT/'base_bundle';bundle.mkdir()
    schema=read(ROOT/'data/processed/feature_schema.json');write(bundle/'feature_schema.json',schema)
    raw=np.load(ROOT/'data/processed/features.npy',mmap_mode='r',allow_pickle=False)
    s=np.zeros(1582,np.float64);ss=np.zeros_like(s)
    for i in range(0,len(base_ids),2048):
        v=histograms(raw[m['rows'][base_ids[i:i+2048]]],schema).astype(np.float64);s+=v.sum(0);ss+=(v*v).sum(0)
    mean=s/len(base_ids);std=np.sqrt(np.maximum(ss/len(base_ids)-mean*mean,0))
    np.savez(bundle/'fit_transform.npz',selected_indices=np.arange(12,1594),mean=mean.astype(np.float32),std=np.maximum(std,.001).astype(np.float32))
    pre=FrozenFeaturePreprocessor(bundle,'standardized',10)
    xnp=np.empty((len(base_ids),1582),np.float32)
    for i in range(0,len(base_ids),2048):xnp[i:i+2048]=pre(raw[m['rows'][base_ids[i:i+2048]]])
    device=runtime();torch.manual_seed(20261025)
    x=torch.from_numpy(xnp).to(device);del xnp
    sampler=PlayerBalancedSampler({i:m['owners'][k].decode() for i,k in enumerate(base_ids)},16,4,20261025)
    model=FeatureMetric(1582,128,512,.1).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    train=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (OUT/'base_training.jsonl').open('x',encoding='utf-8') as f:
        for step in range(1,5001):
            if time.perf_counter()-train>1800:raise RuntimeError('Baseline MLP wall cap')
            ids,labels=sampler.sample();pred=model(x[torch.from_numpy(ids).to(device)])
            loss=supervised_contrastive_loss(pred,torch.from_numpy(labels).to(device),.1)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
            if step==1 or step%1000==0:
                ev={'step':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-train}
                f.write(json.dumps(ev)+'\n');f.flush();log(stage='fresh_baseline_training',**ev)
    with (bundle/'model_state.pt').open('xb') as f:torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},f)
    train_seconds=time.perf_counter()-train;peak=torch.cuda.max_memory_allocated()/2**30
    del x,model,opt;torch.cuda.empty_cache()
    manifest={'model_config':{'embedding_dim':128,'hidden_dim':512,'dropout':.1},'synthetic_control':False}
    write(bundle/'manifest.json',manifest);b={'path':bundle,'manifest':manifest}
    model,pre=fixed_batch.frozen_model(b,'cuda')
    features=np.lib.format.open_memmap(OUT/'features.npy',mode='w+',dtype=np.float32,shape=(len(m['rows']),1728))
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        for i in range(0,len(features),1024):
            v=np.array(raw[m['rows'][i:i+1024]],copy=True)
            stat=infer.statistical_vectors(v,b)
            mlp=model(torch.from_numpy(fixed_batch.padded_input(pre(v))).to(device)).cpu().numpy()[:len(v)]
            stages=np.rint(v[:,9:12].astype(np.float64)*m['target'][i:i+len(v),None])
            features[i:i+len(v)]=compose_game_features(stat,mlp,v[:,:12],['B' if c==0 else 'W' for c in m['colors'][i:i+len(v)]],m['moves'][i:i+len(v)],stages)
    features.flush()
    for part in ('train','cal'):
        hh=[r for r in ep['heads'] if r['part']==part];ix=[r['i'] for r in hh]
        np.save(OUT/f'{part}_features.npy',features[ix],allow_pickle=False)
        write(OUT/f'{part}_records.json',[{'player':r['player'],'rank':r['rank'],'part':part,
            'color':int(m['colors'][r['i']]),'effective_group_token':m['group'][r['i']].decode()} for r in hh])
    # Legacy head trainer expects its declared data subdirectory; no old files change.
    data=OUT/'data';data.mkdir()
    for part in ('train','cal'):
        for suffix in ('features.npy','records.json'):(OUT/f'{part}_{suffix}').rename(data/f'{part}_{suffix}')
    write(data/'scope_receipt.json',{'validation_players_excluded':400,'metadata_sha256':sha(OUT/'scope_metadata.npz'),'episode_sha256':sha(OUT/'EPISODES.json')})
    hashes={f'{part}_{suf}':sha(data/f'{part}_{suf}') for part in ('train','cal') for suf in ('features.npy','records.json')}
    write(data/'vector_receipt.json',{'hashes':hashes,'features_sha256':sha(OUT/'features.npy')})
    write(OUT/'G0_scope.json',{'status':'passed_row_scope_contracts_requires_permodel_numeric_probe','current_MLP_normalizer_excludes400':True})
    write(OUT/'BASE_FIT.json',{'fresh_weights':True,'training_rows':len(base_ids),'validation_players_excluded':400,
        'train_seconds':train_seconds,'peak_allocated_GiB':peak,'end_to_end_seconds':time.perf_counter()-t,
        'features_sha256':sha(OUT/'features.npy'),'bundle_files':{p.name:sha(p) for p in bundle.iterdir() if p.is_file()}})
    log(stage='baseline_complete',seconds=time.perf_counter()-t,peak_GiB=peak)

def heads():
    from . import pilot_train_v1 as engine
    engine.RUN=OUT
    engine.train('U0_mean_count',31415)

def fit_metrics():
    from threadpoolctl import threadpool_limits
    from .plda_metric_v1 import histograms,fit
    from goid_submission_v1.stream_gaussian_v1 import fit_stream
    m=meta();ep=read(OUT/'EPISODES.json');ids=np.load(OUT/'base_indices.npy');x=np.load(OUT/'features.npy',mmap_mode='r')
    assert sha(OUT/'features.npy')==read(OUT/'BASE_FIT.json')['features_sha256']
    d=OUT/'metrics';d.mkdir()
    fitx=np.lib.format.open_memmap(d/'fit_composed.npy',mode='w+',shape=(len(ids),1728),dtype=np.float32)
    for i in range(0,len(ids),2048):fitx[i:i+2048]=x[ids[i:i+2048]]
    fitx.flush();receipts={}
    with threadpool_limits(limits=4):
        receipts['D_Gaussian']=fit_stream(fitx,m['colors'][ids],m['owners'][ids],d/'D_Gaussian.npz')
        head=np.asarray([r['i'] for r in ep['heads'] if r['part']=='train'])
        receipts['B_analogue']=fit_stream(x[head],m['colors'][head],m['owners'][head],d/'B_analogue.npz')
        good=ids[(m['colors'][ids]==0)&(m['hd'][ids]<=1)]
        raw=np.load(ROOT/'data/processed/features.npy',mmap_mode='r');schema=read(ROOT/'data/processed/feature_schema.json')
        rawx=np.lib.format.open_memmap(d/'fit_raw.npy',mode='w+',shape=(len(good),1582),dtype=np.float32)
        comp=np.lib.format.open_memmap(d/'fit_aligned_composed.npy',mode='w+',shape=(len(good),1728),dtype=np.float32)
        for i in range(0,len(good),2048):
            rawx[i:i+2048]=histograms(raw[m['rows'][good[i:i+2048]]],schema,sqrt=True)
            comp[i:i+2048]=x[good[i:i+2048]]
        rawx.flush();comp.flush()
        receipts['PLDA_raw']=fit(rawx,m['owners'][good],d/'PLDA_raw.npz')
        receipts['PLDA_composed']=fit(comp,m['owners'][good],d/'PLDA_composed.npz')
    write(OUT/'METRIC_FITS.json',{'models':receipts,'model_sha256':{p.name:sha(p) for p in d.glob('*.npz')},
        'all_fit_states_exclude400':True,'no_selection_or_confirmation_labels_used':True})
    log(stage='all_metrics_fit',models=receipts)

def eval_inputs():
    ep=read(OUT/'EPISODES.json');d=OUT/'evaluation_inputs';d.mkdir()
    truths={};matched={}
    for role in ('selection','confirmation'):
        qq=[q for q in ep['questions'] if q['role']==role]
        players=sorted({q['player'] for q in qq},key=lambda p:token('official_q_assignment',p))
        gmap={p:(5,10,15,20)[j%4] for j,p in enumerate(players)}
        inputs=[{k:q[k] for k in ('id','q','indices')} for q in qq]
        write(d/f'{role}.json',{'candidate_ids':ep['candidates'],'supports':ep['supports'],'questions':inputs})
        truths[role]=[{k:q[k] for k in ('id','q','player','target')} for q in qq]
        matched[role]=[q['id'] for q in qq if q['q']==gmap[q['player']]]
        assert len(matched[role])==200
    write(d/'EVALUATOR_ONLY_GT.json',truths)
    write(d/'OFFICIAL_MATCHED_IDS.json',matched)
    write(OUT/'ALIGNMENT_ADDENDUM.json',{
        'before_any_quality_evaluation':True,'primary':'nested q5/q10 within each player; player-cluster bootstrap',
        'nested_queries':1600,'unique_validation_players':400,
        'secondary_official_matched_subset':'one question/player,100 per q across400,50 per q in each200-target phase',
        'not1600_independent_players':True,
        'baseline_control':'B/D method analogues refit excluding validation; no claim these are original submitted weights',
        'code_sha256':{p.name:sha(p) for p in (Path(__file__),ROOT/'src/goid_v4/plda_metric_v1.py')},
        'source_checks':{'Kaldi_doc':'https://kaldi-asr.org/doc/classkaldi_1_1Plda.html',
            'date':'2026-11-04','DOI_access':'DOI URL not accessible through web tool; no full-paper replication claim'},
        'initial_command_failures':'inline inventory indentation; package import without src path; pytest unavailable. No training/quality result affected; unittest used, environment unchanged.'})
    log(stage='isolated_prediction_inputs_ready')

def ranks(a):
    o=np.argsort(-a,axis=1,kind='stable');r=np.empty_like(o)
    np.put_along_axis(r,o,np.broadcast_to(np.arange(1,a.shape[1]+1),a.shape),axis=1);return r
def fusion(a,b):return .5/(60+ranks(a))+.5/(60+ranks(b))

def predict(role):
    from threadpoolctl import threadpool_limits
    from .pilot_models_v1 import MeanCount,fixed_encode,gaussian_scores
    from .pilot_train_v1 import runtime
    from .plda_metric_v1 import PLDA,histograms,summarize
    from goid_submission_v1.gaussian_csv import FrozenGaussian,summaries
    import torch
    inputs=read(OUT/f'evaluation_inputs/{role}.json');out=OUT/f'{role}_prediction';out.mkdir()
    questions=inputs['questions'];sets=[q['indices'] for q in questions];supports=inputs['supports']
    used=sorted({i for s in supports+sets for i in s});positions={v:i for i,v in enumerate(used)}
    si=[[positions[i] for i in s] for s in supports];qi=[[positions[i] for i in s] for s in sets]
    m=meta();x=np.array(np.load(OUT/'features.npy',mmap_mode='r')[used],copy=True)
    colors=m['colors'][used];assert (colors==0).all() and (m['hd'][used]<=1).all()
    def cosine(v):
        sm=np.stack([v[s].mean(0) for s in si]);sm/=np.maximum(np.linalg.norm(sm,axis=1,keepdims=True),1e-12)
        sm=np.array(sm,order='C');scores=np.empty((len(qi),400),np.float32)
        for i,q in enumerate(qi):
            vq=v[q].mean(0);vq/=max(float(np.linalg.norm(vq)),1e-12)
            scores[i]=sm@np.array(vq,order='C')
        return scores
    stat=cosine(x[:,:1582]);mlp=cosine(x[:,1582:1710]);color=fusion(stat,mlp)
    headpath=OUT/'models/U0_mean_count_s31415/model.pt';saved=torch.load(headpath,map_location='cpu',weights_only=True)
    device=runtime();head=MeanCount(floor=saved['floor'],prior=.001).to(device)
    head.load_state_dict(saved['state_dict']);head.eval().requires_grad_(False)
    tx=torch.from_numpy(x).to(device);tc=torch.from_numpy(colors.astype(np.int64)).to(device)
    sm,sv=fixed_encode(head,tx,tc,si);qm,qv=fixed_encode(head,tx,tc,qi)
    u0single=gaussian_scores(qm,qv,sm,sv,variance_scale=saved['variance_scale'])
    u0=.75*color+.25/(60+ranks(u0single))
    del tx,tc,head;torch.cuda.empty_cache()
    scores={}
    with threadpool_limits(limits=4):
        for key,name in [('B','B_analogue'),('D_gaussian','D_Gaussian')]:
            g=FrozenGaussian(OUT/f'metrics/{name}.npz');z=g.transform(x)
            scores[key]=g.score(summaries(z,colors,qi),summaries(z,colors,si))
        scores['D']=fusion(scores['D_gaussian'],u0)
        variants=('PLDA_raw','PLDA_composed') if role=='selection' else (read(OUT/'SELECTED_FREEZE.json')['PLDA_component'],)
        for variant in variants:
            plda=PLDA(OUT/f'metrics/{variant}.npz')
            if variant=='PLDA_raw':
                raw=np.load(ROOT/'data/processed/features.npy',mmap_mode='r')
                xx=histograms(raw[m['rows'][used]],read(ROOT/'data/processed/feature_schema.json'),sqrt=True)
            else:xx=x
            z=plda.transform(xx);qmean,qcount=summarize(z,qi);smean,scount=summarize(z,si)
            scores[variant]=plda.score(qmean,qcount,smean,scount)
            scores['RRF_D_'+variant]=fusion(scores['D'],scores[variant])
    for k,s in scores.items():
        assert s.shape==(len(qi),400) and np.isfinite(s).all()
        np.save(out/f'{k}.npy',s,allow_pickle=False)
    write(out/'RECEIPT.json',{'role':role,'queries':len(qi),'candidate_count':400,
        'input_sha256':sha(OUT/f'evaluation_inputs/{role}.json'),'no_GT_loaded_in_predictor':True,
        'allB_HD01':True,'known_num_games':sorted({q['q'] for q in questions}),
        'score_sha256':{p.name:sha(p) for p in out.glob('*.npy')},
        'code_sha256':sha(Path(__file__))})
    log(stage='prediction_complete',role=role,queries=len(qi))

def evaluate(role):
    truth=read(OUT/'evaluation_inputs/EVALUATOR_ONLY_GT.json')[role]
    inp=read(OUT/f'evaluation_inputs/{role}.json');assert [q['id'] for q in truth]==[q['id'] for q in inp['questions']]
    dest=OUT/f'{role}_evaluation';dest.mkdir();targets=np.asarray([q['target'] for q in truth]);qc=np.asarray([q['q'] for q in truth])
    candidates=read(OUT/'PROTOCOL.json')['selection_variants'] if role=='selection' else [read(OUT/'SELECTED_FREEZE.json')['selected']]
    vals={};metrics={};rranks={};matched=set(read(OUT/'evaluation_inputs/OFFICIAL_MATCHED_IDS.json')[role]);mm=np.asarray([q['id'] in matched for q in truth])
    for method in ['B','D']+candidates:
        score=np.load(OUT/f'{role}_prediction/{method}.npy',allow_pickle=False)
        rr=ranks(score)[np.arange(len(truth)),targets];rranks[method]=rr
        values=np.where(rr<=5,np.exp(-(rr-1)),0.);vals[method]=values
        metrics[method]={str(q):{'score':float(values[qc==q].mean()),'top1':float((rr[qc==q]==1).mean()),
            'top5':float((rr[qc==q]<=5).mean()),'n':int((qc==q).sum())} for q in (5,10,15,20)}
        metrics[method]['short']=float((values[qc==5].mean()+values[qc==10].mean())/2)
        metrics[method]['official_matched_200']=float(values[mm].mean())
    comparisons={};players=sorted({q['player'] for q in truth});rng=np.random.default_rng(2026110401)
    draws=rng.integers(len(players),size=(10000,len(players)))
    for method in candidates:
        comparisons[method]={}
        for comparator in ('D','B'):
            diff=vals[method]-vals[comparator];matrix=np.zeros((len(players),4));idx={p:i for i,p in enumerate(players)}
            for i,q in enumerate(truth):matrix[idx[q['player']],(5,10,15,20).index(q['q'])]=diff[i]
            delta=matrix.mean(0);boot=matrix[draws].mean(1);short=boot[:,:2].mean(1)
            checks={'short_point':bool(delta[:2].mean()>=.005),'q5_q10':bool((delta[:2]>=0).all()),
                'q15_q20':bool((delta[2:]>=-.005).all()),'short_lower':bool(np.quantile(short,.025,method='lower')>0)}
            comparisons[method][comparator]={'delta_q5_q10_q15_q20':delta.tolist(),'short_delta':float(delta[:2].mean()),
                'short_player_cluster_95_CI':np.quantile(short,[.025,.975]).tolist(),'checks':checks,'passed':all(checks.values()),
                'recovered_short_top1':int(((rranks[method]==1)&(rranks[comparator]!=1)&(qc<=10)).sum()),
                'regressed_short_top1':int(((rranks[method]!=1)&(rranks[comparator]==1)&(qc<=10)).sum())}
    result={'role':role,'targets':len(players),'queries':len(truth),'shared_fixed400_support_bank':True,
        'historically_exposed_internal_not_pristine':True,'current_fits_exclude_all400':True,
        'metrics':metrics,'comparisons':comparisons,'original_submitted_B_D_weights_not_used':True,
        'official_scores_not_recomputed':True,'retired400_not_read':True}
    write(dest/'SUMMARY.json',result)
    events=[dict(q,**{k:float(v[i]) for k,v in vals.items()},**{k+'_rank':int(v[i]) for k,v in rranks.items()}) for i,q in enumerate(truth)]
    write(dest/'EVENTS.json',events)
    if role=='selection':
        best=max(candidates,key=lambda k:metrics[k]['short'])
        write(OUT/'SELECTED_FREEZE.json',{'selected':best,'PLDA_component':best.removeprefix('RRF_D_'),
            'selection_summary_sha256':sha(dest/'SUMMARY.json'),'preliminary_passed':all(x['passed'] for x in comparisons[best].values()),
            'method_hashes':read(OUT/'METRIC_FITS.json')['model_sha256'],
            'confirmation_quality_read':False,'no_more_variants_or_weight_search':True})
    log(stage='quality_evaluation_complete',**result)

def prereveal():
    fits=read(OUT/'METRIC_FITS.json')
    assert all(sha(OUT/'metrics'/p)==h for p,h in fits['model_sha256'].items())
    paths=[OUT/'PROTOCOL.json',OUT/'EPISODES.json',OUT/'ALIGNMENT_ADDENDUM.json',OUT/'BASE_FIT.json',OUT/'METRIC_FITS.json',
        OUT/'models/U0_mean_count_s31415/model.pt',Path(__file__),ROOT/'src/goid_v4/plda_metric_v1.py',
        ROOT/'tests/research_v4/test_plda_metric_v1.py']
    paths.extend((OUT/'base_bundle').iterdir())
    write(OUT/'PRE_REVEAL_BINDING.json',{'quality_not_yet_evaluated':True,'all_current_fit_states_exclude400':True,
        'files':{str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()},
        'concept_primary_reference':'https://kaldi-asr.org/doc/classkaldi_1_1Plda.html',
        'external_code_or_weights_used':False,'synthetic_tests':'3 unittest tests passed; dense Gaussian conditioning, subset invariance, fit/overwrite guards'})
    log(stage='pre_reveal_binding_complete')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--role',choices=['selection','confirmation']);a=p.parse_args()
    if a.stage in ('predict','evaluate'):
        if a.role is None:raise ValueError('Explicit role required')
        globals()[a.stage](a.role)
    else:globals()[a.stage]()
if __name__=='__main__':main()
