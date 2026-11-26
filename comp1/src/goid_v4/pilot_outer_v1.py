"""Prospective blocked INTERNAL U0 test; all learned states refit without800 targets.

The targets were original MLP fit identities, not original pristine players.
No retired400 files are opened. The unchanged head trainer is parameterized to
a new run root in the current process, never modifying its source or old data.
"""
from __future__ import annotations
from collections import defaultdict,Counter
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .pilot_scope_v1 import ROOT,PointReader,sha,write,fit_player,opaque,RANKS
from .artifacts import ContractError,object_hash

OUT=ROOT/'runs/research_v4/p1_u0_outer_v1_20261026'
PREV=ROOT/'runs/research_v4/p1_pilot_v1_20261026'


def prepare():
    start=time.perf_counter();d=OUT/'data';d.mkdir(exist_ok=False)
    config=json.loads((ROOT/'configs/research_v4/p1_u0_outer_v1_20261026.json').read_text(encoding='utf-8'))
    head={}
    for part in ('train','cal'):
        for row in json.loads((PREV/f'data/{part}_records.json').read_text(encoding='utf-8')):head[row['row_id']]=part
    old_players={r['player'] for part in ('train','cal') for r in json.loads((PREV/f'data/{part}_records.json').read_text(encoding='utf-8'))}
    path=ROOT/'runs/statistical_v1/fit_row_ids.npy'
    if sha(path)!='b0bf9c6a91317456d53ff63555dea5a2249041f7538de058e41ce81d9cd30bd8':raise ContractError('Original fit index changed')
    ids=np.load(path,allow_pickle=False)
    reader=PointReader(ROOT/'data/processed/audit.sqlite')
    try:rows=reader.fetch(ids);query_plan=reader.plan
    finally:reader.close()
    by_player=defaultdict(list)
    for r in rows:
        if not fit_player(r['player_id']):raise ContractError('Original fit identity boundary failed')
        by_player[r['player_id']].append(r)
    by_rank=defaultdict(list)
    for p,rr in by_player.items():
        if opaque(p) not in old_players and len(rr)>=120 and len({r['rank'] for r in rr})==1:
            by_rank[rr[0]['rank']].append(p)
    quotas={rank:min(40,len(by_rank[rank])//2) for rank in RANKS}
    while sum(quotas.values())<400:
        before=sum(quotas.values())
        for rank in RANKS:
            if 2*(quotas[rank]+1)<=len(by_rank[rank]):quotas[rank]+=1
            if sum(quotas.values())==400:break
        if sum(quotas.values())==before:raise ContractError('Insufficient two-pool prospective capacity')
    outer={};selected_by_pool=[[],[]]
    for rank in RANKS:
        players=sorted(by_rank[rank],key=lambda p:object_hash({'outer_seed':20261026,'player':p}))
        for pool in (0,1):
            for p in players[pool*quotas[rank]:(pool+1)*quotas[rank]]:
                outer[p]=pool;rr=sorted(by_player[p],key=lambda r:object_hash({'seed':20261026,'group':r['group_key']}))[:120]
                selected_by_pool[pool].extend(rr)
    if len(outer)!=800 or {opaque(p) for p in outer}&old_players:raise ContractError('Outer exposure selection failed')
    base=[r for r in rows if r['player_id'] not in outer]
    allouter=[r for r in rows if r['player_id'] in outer]
    for k in ('raw_hash','move_hash','canonical_hash','group_key'):
        if {r[k] for r in base}&{r[k] for r in allouter}:raise ContractError(f'Background/outer known {k} overlap')
    selected=base+selected_by_pool[0]+selected_by_pool[1]
    selected.sort(key=lambda r:r['row_id'])
    raw=np.load(ROOT/'data/processed/features.npy',mmap_mode='r',allow_pickle=False)
    matrix=np.lib.format.open_memmap(d/'source_raw_features.npy',mode='w+',dtype=np.float32,shape=(len(selected),1594))
    for i in range(0,len(selected),2048):matrix[i:i+2048]=raw[[r['row_id'] for r in selected[i:i+2048]]]
    matrix.flush();del matrix,raw
    base_ids=np.asarray([i for i,r in enumerate(selected) if r['player_id'] not in outer],np.int64)
    np.save(d/'base_indices.npy',base_ids,allow_pickle=False)
    full=[]
    parser_sha=sha(ROOT/'src/goid/sgf_features.py')
    for r in selected:
        key=hashlib.sha256(f"goid-feature-cache-v1|{parser_sha}|1.1.1|{r['raw_hash']}|{r['color']}".encode()).hexdigest()
        full.append({'row_id':r['row_id'],'key':key,'player':opaque(r['player_id']),'rank':r['rank'],'color':0 if r['color']=='B' else 1,
                     'head_part':head.get(r['row_id']),'outer_pool':outer.get(r['player_id']),
                     'group':r['group_key'],'raw_hash':r['raw_hash'],'move_hash':r['move_hash'],'canonical_hash':r['canonical_hash'],
                     'mainline_moves':r['mainline_moves'],'target_moves':r['target_moves']})
    write(d/'source_records.json',full)
    used=sorted([i for i,r in enumerate(full) if r['head_part'] is not None or r['outer_pool'] is not None],key=lambda i:full[i]['key'])
    # Only selected120 games per outer player were exported, all other outer games stay excluded.
    if len(used)!=156000:raise ContractError('Expected60000 fit-head/cal and96000 outer game rows')
    np.save(d/'used_indices.npy',np.asarray(used,np.int64),allow_pickle=False)
    records=[dict(full[i],part=full[i]['head_part'] if full[i]['head_part'] else f"outer{full[i]['outer_pool']}") for i in used]
    pos={r['row_id']:i for i,r in enumerate(records)}
    pools=[]
    for pool in (0,1):
        players=sorted([p for p,v in outer.items() if v==pool],key=lambda p:object_hash({'candidate_order':p,'seed':20261026}))
        supports=[];questions=[]
        for j,p in enumerate(players):
            rr=[r for r in selected_by_pool[pool] if r['player_id']==p]
            # Preserve the preselected hash order for supports/query prefix.
            supports.append([pos[r['row_id']] for r in rr[:100]])
            for q in (5,10,20):questions.append({'question_id':f'outer{pool}_p{j:04}_q{q}','player':opaque(p),'q':q,'indices':[pos[r['row_id']] for r in rr[100:100+q]]})
        pools.append({'name':f'outer{pool}','candidate_order':[opaque(p) for p in players],'supports':supports,'questions':questions})
    write(d/'allowed_records.json',records);write(d/'outer_plans.json',pools)
    receipt={'status':'prepared_prospective_blocked_internal','original_fit_rows':len(ids),'base_fit_rows':len(base_ids),'base_fit_players':len(by_player)-800,
             'head_train_players':400,'head_cal_players':100,'outer_players':800,'outer_pools':2,'quota_per_pool':quotas,
             'outer_previously_P1_head_or_cal':0,'outer_previously_original_MLP_fit':800,'old_original_MLP_weights_will_not_be_loaded':True,
             'background_vs_outer_player_and_known_group_overlap':0,'query_plan':query_plan,'retired_rows_or_names_read':False,
             'scope_hash':object_hash([(r['row_id'],r['part']) for r in records]),'source_raw_features_sha256':sha(d/'source_raw_features.npy'),
             'allowed_records_sha256':sha(d/'allowed_records.json'),'protocol_sha256':sha(ROOT/'configs/research_v4/p1_u0_outer_v1_20261026.json'),
             'source_code_sha256':sha(Path(__file__)),'elapsed_seconds':time.perf_counter()-start,'pristine_independent_confirmation':False}
    write(d/'scope_receipt.json',receipt)
    print(json.dumps({'stage':'outer_scope_prepared',**receipt},ensure_ascii=False),flush=True)


def hist_block(raw,schema):
    x=np.asarray(raw[:,12:],np.float32).copy();offset=0
    for g in schema['groups']:
        if g['kind']!='histogram':continue
        width=g['stop']-g['start'];v=x[:,offset:offset+width]
        v/=np.maximum(v.sum(1,keepdims=True),np.float32(1e-12));offset+=width
    if offset!=1582:raise ContractError('Histogram dimension changed')
    return x


def base():
    import os
    from .pilot_train_v1 import runtime
    from goid.feature_metric import FeatureMetric,PlayerBalancedSampler,supervised_contrastive_loss,FrozenFeaturePreprocessor
    from goid.fixed_batch import encode_features
    from .p1 import compose_game_features
    start=time.perf_counter();d=OUT/'data';bundle=OUT/'base_bundle';bundle.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'configs/research_v4/p1_u0_outer_v1_20261026.json').read_text(encoding='utf-8'))
    schema=json.loads((ROOT/'artifacts/inference_bundle_v1/feature_schema.json').read_text(encoding='utf-8'))
    write(bundle/'feature_schema.json',schema)
    rows=json.loads((d/'source_records.json').read_text(encoding='utf-8'));raw=np.load(d/'source_raw_features.npy',mmap_mode='r',allow_pickle=False)
    base_ids=np.load(d/'base_indices.npy',allow_pickle=False);s=np.zeros(1582,np.float64);ss=np.zeros_like(s)
    for i in range(0,len(base_ids),4096):
        block=hist_block(raw[base_ids[i:i+4096]],schema).astype(np.float64);s+=block.sum(0);ss+=(block*block).sum(0)
    mean=s/len(base_ids);std=np.sqrt(np.maximum(ss/len(base_ids)-mean*mean,0))
    np.savez(bundle/'fit_transform.npz',selected_indices=np.arange(12,1594),mean=mean.astype(np.float32),std=np.maximum(std,.001).astype(np.float32))
    pre=FrozenFeaturePreprocessor(bundle,'standardized',10)
    normalized=np.lib.format.open_memmap(d/'base_preprocessed.npy',mode='w+',dtype=np.float32,shape=(len(base_ids),1582))
    for i in range(0,len(base_ids),2048):normalized[i:i+2048]=pre(raw[base_ids[i:i+2048]])
    normalized.flush();device=runtime();torch.manual_seed(20261025)
    x=torch.from_numpy(np.array(normalized,copy=True)).to(device)
    owners={i:rows[int(k)]['player'] for i,k in enumerate(base_ids)}
    sampler=PlayerBalancedSampler(owners,16,4,20261025)
    model=FeatureMetric(1582,128,512,.1).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    train_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (OUT/'base_training.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,5001):
            if time.perf_counter()-train_start>cfg['MLP_wall_cap_seconds']:raise ContractError('Base refit wall cap')
            ids,labels=sampler.sample();z=model(x[torch.from_numpy(ids).to(device)])
            loss=supervised_contrastive_loss(z,torch.from_numpy(labels).to(device),.1)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
            if step==1 or step%1000==0:
                row={'step':step,'loss':float(loss.detach()),'elapsed_seconds':time.perf_counter()-train_start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps({'stage':'baseline_refit',**row}),flush=True)
    with (bundle/'model_state.pt').open('xb') as f:torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},f)
    base_seconds=time.perf_counter()-train_start
    del model,optimizer,x,normalized;torch.cuda.empty_cache()
    manifest={'synthetic_control':False,'model_config':{'embedding_dim':128,'hidden_dim':512,'dropout':.1},'parser_sha256':sha(ROOT/'src/goid/sgf_features.py'),
              'files':{p:sha(bundle/p) for p in ('model_state.pt','fit_transform.npz','feature_schema.json')},'all_parameters_fresh_fit_excluding_outer':True}
    write(bundle/'manifest.json',manifest)
    used=np.load(d/'used_indices.npy',allow_pickle=False)
    selected=np.array(raw[used],copy=True);(stat,mlp),info=encode_features(selected,{'manifest':manifest,'path':bundle},device='cuda')
    records=json.loads((d/'allowed_records.json').read_text(encoding='utf-8'))
    counts=np.asarray([r['target_moves'] for r in records],float)
    stages=np.rint(selected[:,9:12].astype(np.float64)*counts[:,None])
    features=compose_game_features(stat,mlp,selected[:,:12],['B' if r['color']==0 else 'W' for r in records],[r['mainline_moves'] for r in records],stages)
    np.save(d/'features.npy',features,allow_pickle=False);np.save(d/'stat.npy',stat,allow_pickle=False);np.save(d/'mlp.npy',mlp,allow_pickle=False)
    for part in ('train','cal'):
        idx=[i for i,r in enumerate(records) if r['part']==part]
        np.save(d/f'{part}_features.npy',features[idx],allow_pickle=False);write(d/f'{part}_records.json',[records[i] for i in idx])
    names=['features.npy','stat.npy','mlp.npy','train_features.npy','train_records.json','cal_features.npy','cal_records.json']
    write(d/'vector_receipt.json',{'hashes':{name:sha(d/name) for name in names},'baseline_encoder':info,'old_weights_loaded':False,'normalizer_fit_rows':len(base_ids),'outer_normalizer_or_MLP_fit_rows':0})
    write(OUT/'G0_scope.json',{'status':'passed_row_scope_contracts_requires_permodel_numeric_probe','outer_players':800,'head_fit_vs_outer_player_groups_disjoint':True,
           'every_learned_baseline_reset_and_refit':True,'current_MLP_or_normalizer_outer_fit_rows':0,'scope_receipt_sha256':sha(d/'scope_receipt.json'),
           'outer_protocol_sha256':sha(ROOT/'configs/research_v4/p1_u0_outer_v1_20261026.json'),'original_retired400_read':False,'route':'prospective_blocked_internal'})
    write(OUT/'baseline_refit_result.json',{'steps':5000,'train_seconds':base_seconds,'end_to_end_prepare_refit_encode_seconds':time.perf_counter()-start,
          'model_sha256':sha(bundle/'model_state.pt'),'fit_transform_sha256':sha(bundle/'fit_transform.npz'),'outer_players_excluded':800,'old_weights_loaded':False})
    print(json.dumps({'stage':'fresh_baseline_and_vectors_complete','elapsed_seconds':time.perf_counter()-start}),flush=True)


def heads():
    from . import pilot_train_v1 as engine
    previous=engine.RUN;engine.RUN=OUT
    write(OUT/'head_run_root_parameterization.json',{'original_root':str(previous),'actual_root':str(OUT),'trainer_source_sha256':sha(ROOT/'src/goid_v4/pilot_train_v1.py'),
          'outer_protocol_sha256':sha(ROOT/'configs/research_v4/p1_u0_outer_v1_20261026.json'),'training_hyperparameters_unchanged':True,'old_source_modified':False})
    for seed in (31415,27182,16180):engine.train('U0_mean_count',seed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','base','heads']);a=p.parse_args();globals()[a.stage]()
