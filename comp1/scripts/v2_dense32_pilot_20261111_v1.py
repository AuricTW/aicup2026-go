"""Original-train-only, B/W nested8/32-decision few-shot pilot.

Keeps all previous source/models/runs immutable; no upload or Public-key code.
Staged entrypoints freeze identities, episodes and gates before new quality.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import sys,json,hashlib,time,re,argparse,multiprocessing
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from threadpoolctl import threadpool_limits
import build_v2_baseline_20261111_v2 as b
import v2_color_plda_20261111_v1 as cp
from goid_kg_v1 import native as n
from goid_kg_v1.features import aggregate
from goid_v4.plda_metric_v1 import PLDA,fit as fit_plda,summarize
RUN=ROOT/'runs/v2_score_push_20261111_v1'
DENSE=ROOT/'runs/nearperfect_local_pilot_20261105_v1/dense64'
SEED=2026111121

def select(moves,target,canonical,quota=16):
    choices=[[],[]]
    for ply,(color,_) in enumerate(moves,1):
        if color==target.lower() and ply<=180:
            h=hashlib.sha256(f'{canonical}|{target.lower()}|{ply}|20261104'.encode()).digest()
            choices[int(ply>60)].append((h,ply))
    return {p for x in choices for _,p in sorted(x)[:quota]}

def prepare():
    m=b.meta();eps=[b.read(b.RUN/f'calibration/ep{i}.json') for i in range(2)]
    valowners=set(eps[0]['candidates']);blocked=valowners|set(b.read(ROOT/'runs/4090_support100_pilot_20261105_v1/EPISODE_A.json')['candidates'])
    excluded=np.isin(m['owners'].astype(str),sorted(blocked));safe=(~excluded)&(m['hd']<=1)
    for f in b.FAMILIES:safe &= ~np.isin(m[f],np.unique(m[f][excluded]))
    oldtrain=np.load(DENSE.parent/'train_indices.npy');oldset=set(np.load(DENSE/'indices.npy').tolist())
    by=defaultdict(lambda:[[],[]])
    for i in oldtrain:
        if safe[i]:by[m['owners'][i].decode()][0].append(int(i));assert int(i) in oldset
    for i in np.flatnonzero(safe & (m['colors']==1)):
        p=m['owners'][i].decode()
        if p in by:by[p][1].append(int(i))
    eligible=[p for p,cc in by.items() if min(map(len,cc))>=16]
    # Stable rank-balanced round robin; metadata only, no embeddings/quality.
    rankgroups=defaultdict(list)
    for p in eligible:rankgroups[str(m['ranks'][by[p][0][0]])].append(p)
    for r in rankgroups:rankgroups[r].sort(key=lambda p:hashlib.sha256((f'{SEED}|'+p).encode()).digest())
    players=[];cursor=0
    while len(players)<512:
        changed=False
        for r in sorted(rankgroups):
            if cursor<len(rankgroups[r]):players.append(rankgroups[r][cursor]);changed=True
            if len(players)==512:break
        assert changed,'Insufficient mixed-color training identities'
        cursor+=1
    train=[];labels=[]
    for j,p in enumerate(players):
        for c in (0,1):
            ix=sorted(by[p][c],key=lambda i:hashlib.sha256(m['group'][i]+str(SEED).encode()).digest())[:16]
            train+=ix;labels += [j]*16
    train=np.asarray(train,np.int64);labels=np.asarray(labels,np.int64)
    assert len(train)==len(np.unique(train))==16384 and len(set(m['owners'][train]))==512
    assert not set(m['owners'][train].astype(str))&blocked
    # Fresh query games on the fixed ep0 support bank. Same previously exposed
    # identities: this is fresh-question conditional validation, not new people.
    ss=eps[0]['supports'];support=np.concatenate(ss);sf={f:set(m[f][support]) for f in b.FAMILIES}
    oldq=np.asarray(sorted({i for e in eps for q in e['questions'] for i in q['indices']}))
    usedq={f:set(m[f][oldq]) for f in b.FAMILIES};positives=sorted({q['player'] for e in eps for q in e['questions']})
    qp=defaultdict(list)
    for i in np.flatnonzero(np.isin(m['owners'].astype(str),positives)&(m['hd']<=1)):
        if all(m[f][i] not in sf[f] and m[f][i] not in usedq[f] for f in b.FAMILIES):qp[m['owners'][i].decode()].append(int(i))
    assert all(len(qp[p])>=5 for p in positives)
    rng=np.random.default_rng(SEED);qs=[]
    for count in (1,2,3,5):
        for p in rng.choice(positives,100,replace=False):
            ids=sorted(rng.choice(qp[p],count,replace=False).tolist(),key=lambda i:m['raw'][i])
            qs.append({'id':f'fresh_{count}_{len(qs):04}','indices':ids,'target':eps[0]['candidates'].index(p),'player':str(p),'q':count})
    fresh={'seed':SEED,'candidates':eps[0]['candidates'],'supports':ss,'questions':qs,'same_player_fixed_support_conditional_confirmation':True}
    b.write(RUN/'EPISODE_CONFIRMATION.json',fresh)
    cal=np.asarray(sorted({i for e in eps+[fresh] for s in e['supports'] for i in s}|{i for e in eps+[fresh] for q in e['questions'] for i in q['indices']}),np.int64)
    assert not set(train)&set(cal)
    for f in b.FAMILIES:assert not set(m[f][train])&set(m[f][cal]),f
    for name,arr in [('train_indices',train),('train_labels',labels),('cal_indices',cal)]:np.save(RUN/f'{name}.npy',arr,allow_pickle=False)
    selected=set(train)|set(cal)
    cached=np.asarray(sorted(i for i in train if int(i) in oldset and m['colors'][i]==0),np.int64)
    pending=np.asarray(sorted(selected-set(cached.tolist())),np.int64)
    np.save(RUN/'cached_B_indices.npy',cached,allow_pickle=False);np.save(RUN/'replay_indices.npy',pending,allow_pickle=False)
    b.write(RUN/'PROTOCOL.json',{'created_utc':b.base.utc(),'version':'v2_dense32_pilot_20261111_v1','baseline':'F_C_H25','baseline_run':str(b.RUN),
      'official_test_labels_not_available':True,
      'training_players':512,'training_games':16384,'games_per_player':32,'training_colors_per_player':{'B':16,'W':16},
      'training_excludes_all600_candidates_and_current_A_B_owner_and_raw_move_canonical_group':True,
      'source_safe_SHA256':b.sha(b.SAFE),'retired400_excluded_by_safe_source':True,'fresh_episode_queries':400,'fresh_query_families_disjoint_from_both_screen_query_banks':True,
      'confirmation_reuses_players_and_ep0_support_not_independent':True,
      'features':'frozen HumanSL,rank_1d,actual B/W,HD0/1,nested hash4+4 vs16+16 decisions,none afterply180',
      'new_selector_not_limited_by_old16_decision_guard':'new isolated implementation, original selector unchanged',
      'training_B_reuse':'audited prior dense64 tokens -> nested8/32 summaries; float16 token compression on all colors/views',
      'scaler':'fit train only; clip10, zero constant columns; separate8/32',
      'models':['raw_PLDA128_8','raw_PLDA128_32','episodic10support_PLDA128_8','episodic10support_PLDA128_32'],
      'episodic_head':{'network':'2339->512 LayerNorm GELU Dropout0.1 ->128 L2','steps':2000,'seed':SEED,'optimizer':'AdamW','lr':.0003,'weight_decay':.01,'clip':5,'temperature':.07,'candidates_per_step':512,'supports_per_candidate':10,'query_players_per_step':64,'query_counts':[1,2,3,5],'query_support_disjoint_games':True,'checkpoint':'last','same_draws_for8_and32':True},
      'fusion':'(1-w)*baseline_F_C_H25+w/(60+ranknew)','weights':[.1,.25,.4],
      'screen_gate':{'mean_delta_min':.01,'each_episode_delta_min':0,'q1_q2_delta_min':0,'q3_q5_delta_floor':-.003,'top1_net_min':8,'shared_player_cluster95_lower_gt':0,'draws':5000,'seed':SEED},
      'selection':'highest mean passing score, tie smallerw thenrawPLDA then8; evaluate one frozen winner on fresh episode only',
      'confirmation_gate':{'mean_delta_min':.005,'q1_q2_delta_min':0,'q3_q5_delta_floor':-.005,'top1_net_min':0,'player_cluster95_lower_gt':0},
      'if_fail':'retain baseline; no failed pilot in official predictions','old800':'exact unchanged frozen H predictions','no_upload':True,
      'compute':'4090 only after fresh capacity and ownership check; otherwise local3060','feature_GPU_batch':32,'game_partition':'each game fixed32 duplicate-first padding, independent of other games','model_GPU_batch':128,'TF32':False,
      'cached_B_games':len(cached),'new_replay_games':len(pending),'cal_games':len(cal),'caps_seconds':{'features':7200,'head_each':1200,'official_features':3600},
      'original_source_SHA256':b.sha(__file__)})
    b.write(RUN/'SCOPE_AUDIT.json',{'eligible_train_players':len(eligible),'train_players':512,'train_games':len(train),'calibration_candidates':600,'positive_identities':len(positives),'cal_unique_games':len(cal),'cached_B':len(cached),'new_replay':len(pending),'all_owner_and_four_families_train_cal_overlap':0,'Public_Private_B_retired_details_loaded':False})
    b.log(stage='dense32_prepared',train=len(train),cal=len(cal),cached=len(cached),new_replay=len(pending))

def encode(item):
    index,text,expected,hd,target,config=item
    from katago.game.board import Board
    from katago.game.features import Features
    from katago.game.gamestate import GameState
    from goid.sgf_features import analyze_sgf,decode_sequence
    assert hashlib.sha256(text.encode()).hexdigest()==expected and hd in (0,1) and target in ('B','W')
    p=analyze_sgf(text,target,include_features=False);moves=decode_sequence(p['move_sequence'],19);assert not decode_sequence(p['setup_sequence'],19)
    chosen=select(moves,target,p['canonical_hash']);small=select(moves,target,p['canonical_hash'],4);assert small<=chosen
    if not chosen:return index,None,'no_target_decision'
    rules=GameState.RULES_JAPANESE.copy();rules['whiteKomi']=6.5 if hd==0 else 0.;gs=GameState(19,rules);fe=Features(config,19)
    meta=n.profile_metadata('rank_1d').get_metadata_row(target,361);px=[];gg=[];ll=[];aa=[];aux=[];mask=[];plies=[];previous=False;double=False
    try:
        for ply,(color,point) in enumerate(moves,1):
            pla=Board.BLACK if color=='b' else Board.WHITE;loc=Board.PASS_LOC if point is None else gs.board.loc(point[1],18-point[0])
            if ply in chosen:
                assert not double and color==target.lower() and gs.board.pla==pla
                gs.rules['passWouldEndPhase']=previous;x,g=gs.get_input_features(fe)
                if hd==1:g[0,5]+=(.5 if target=='W' else -.5)/20
                legal=np.asarray([gs.board.would_be_legal(pla,fe.tensor_pos_to_loc(i,gs.board)) and not gs.board.would_be_suicide(pla,fe.tensor_pos_to_loc(i,gs.board)) for i in range(361)]+[True])
                action=361 if point is None else (18-point[0])*19+point[1];assert legal[action]
                assert ((x==0)|(x==1)).all() and np.isfinite(g).all()
                px.append(np.packbits(x[0].reshape(-1).astype(np.uint8)));gg.append(g[0]);ll.append(np.packbits(legal));aa.append(action);aux.append([float(ply<=60),float(ply>60),min(ply/180,1),float(hd),float(action==361)]);mask.append(ply in small);plies.append(ply)
            if ply>=max(chosen):break
            gs.play(pla,loc);double=double or(previous and point is None);previous=point is None
    except Exception as e:return index,None,f'{type(e).__name__}:{e}'
    return index,(np.stack(px),np.stack(gg),np.stack(ll),np.asarray(aa,np.int64),np.asarray(aux,np.float32),meta,np.asarray(mask,bool),np.asarray(plies,np.int16)),None

def encode_batch(items):return [encode(x) for x in items]

def infer(model,r):
    import torch
    assert r[1] is not None and r[2] is None,(r[0],r[2])
    px,glob,legal,actions,aux,meta,small,plies=r[1];count=len(actions);assert 1<=count<=32
    ix=np.pad(np.arange(count),(0,32-count),constant_values=0);device=next(model.parameters()).device
    x=np.unpackbits(px[ix],axis=1)[:,:22*361].reshape(32,22,19,19).astype(np.float32);mask=np.unpackbits(legal[ix],axis=1)[:,:362].astype(bool)
    pi,h=n.forward(model,torch.from_numpy(x).to(device),torch.from_numpy(glob[ix]).to(device),torch.from_numpy(np.repeat(meta[None],32,axis=0)).to(device),torch.from_numpy(mask).to(device),amp=True)
    h=h.float().flatten(2);ac=torch.from_numpy(actions[ix]).to(device);local=h.gather(2,ac.clamp_max(360)[:,None,None].expand(-1,384,1)).squeeze(2);local[ac==361]=0
    expected=(h*pi[:,:361,None].transpose(1,2)).sum(2);z=torch.cat((local,h.mean(2),local-expected),1).cpu().numpy()[:count]
    chosen=pi.gather(1,ac[:,None]).squeeze(1);rank=((pi>chosen[:,None])&torch.from_numpy(mask).to(device)).sum(1).float()/torch.from_numpy(mask.sum(1)).to(device)
    entropy=-(pi*torch.log(pi.clamp_min(1e-12))).sum(1);top=pi.topk(2,dim=1).values
    numbers=torch.stack((torch.log(chosen.clamp_min(1e-12)),rank,entropy,torch.log(top[:,0].clamp_min(1e-12)),top[:,0]-top[:,1]),1).cpu().numpy()[:count]
    a=np.concatenate((aux[:,:2],numbers,aux[:,2:]),1)
    # Match the existing B training cache's token storage precision, on W too.
    z=z.astype(np.float16).astype(np.float32)
    return aggregate(z[small],a[small]),aggregate(z,a),{'tokens':z,'aux':a,'small':small,'plies':plies}

def reuse_B():
    m=b.meta();ii=np.load(DENSE/'indices.npy');positions={int(i):j for j,i in enumerate(ii)}
    tokens=np.load(DENSE/'tokens.npy',mmap_mode='r');aux=np.load(DENSE/'aux.npy',mmap_mode='r');plies=np.load(DENSE/'plies.npy',mmap_mode='r');counts=np.load(DENSE/'counts.npy')
    dest=RUN/'cached_B';dest.mkdir();ids=np.load(RUN/'cached_B_indices.npy');out={v:np.empty((len(ids),2339),np.float32) for v in (8,32)}
    receipt=b.read(DENSE/'RECEIPT.json')
    for file in ('indices.npy','plies.npy','counts.npy','aux.npy','tokens.npy'):assert b.sha(DENSE/file)==receipt['files'][file],file
    for j,i in enumerate(ids):
        k=positions[int(i)];pp=plies[k,:counts[k]];pools=[[],[]]
        for a,p in enumerate(pp):pools[int(p>60)].append((hashlib.sha256(f'{m["canonical"][i].decode()}|b|{int(p)}|20261104'.encode()).digest(),a))
        for view,quota in ((8,4),(32,16)):
            selected=sorted(a for pool in pools for _,a in sorted(pool)[:quota]);out[view][j]=aggregate(tokens[k,selected].astype(np.float32),aux[k,selected])
    for view,value in out.items():np.save(dest/f'view{view}.npy',value,allow_pickle=False)
    b.write(dest/'RECEIPT.json',{'games':len(ids),'views':[8,32],'token_precision':'float16 from preserved dense64 cache','source_receipt_SHA256':b.sha(DENSE/'RECEIPT.json'),'no_new_backbone_forward':True})
    b.log(stage='cached_B_ready',games=len(ids))

def tasks(indices,official=False):
    config=b.read(n.RUN/'model_config.json')
    if official:
        games,_=b.load_official();keys=b.read(b.RUN/'official_stat/keys.json')
        for start in range(0,len(keys),32):
            yield [(j,games[k].sgf,games[k].raw_hash,int(re.match(r'^\(;SZ\[19\]HD\[([01])\];',games[k].sgf).group(1)),games[k].color,config) for j,k in enumerate(keys[start:start+32],start)]
    else:
        m=b.meta();from goid_v4.pilot_scope_v1 import PointReader
        reader=PointReader(ROOT/'data/processed/audit.sqlite')
        try:
            for start in range(0,len(indices),32):
                ids=indices[start:start+32];texts={r['row_id']:r['sgf_content'] for r in reader.sample_text(m['rows'][ids])}
                yield [(int(i),texts[int(m['rows'][i])],m['raw'][i].decode(),int(m['hd'][i]),'BW'[m['colors'][i]],config) for i in ids]
        finally:reader.close()

def extract(official=False):
    b.runtime();dest=RUN/('official32' if official else 'replayed');dest.mkdir(exist_ok=False)
    ids=np.arange(len(b.read(b.RUN/'official_stat/keys.json'))) if official else np.load(RUN/'replay_indices.npy');positions={int(i):j for j,i in enumerate(ids)}
    arrays={v:np.lib.format.open_memmap(dest/f'view{v}.npy',mode='w+',dtype=np.float32,shape=(len(ids),2339)) for v in (8,32)}
    start=time.perf_counter();model=n.load_backbone('cuda',False);done=np.zeros(len(ids),bool)
    iterator=iter(tasks(ids,official));first=next(iterator);probe=[encode(x) for x in first[:4]];cold={r[0]:infer(model,r)[:2] for r in probe};warm={r[0]:infer(model,r)[:2] for r in reversed(probe)}
    assert all(np.array_equal(cold[k][v],warm[k][v]) for k in cold for v in (0,1))
    for item,r in zip(first[:4],probe):
        old=b.encode_target(item);assert old[2] is None
        for field in range(5):assert np.array_equal(r[1][field][r[1][6]],old[1][field])
    b.write(dest/'ENGINEERING.json',{'nested8_arrays_exact_vs_original_BW':True,'cold_warm_subset_reverse_exact':True,'fixed32':True,'views_share_same_decisions_and_quantization':True})
    import itertools
    batches=iter(itertools.chain([first],iterator))
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending=set()
        for _ in range(8):
            batch=next(batches,None)
            if batch:pending.add(pool.submit(encode_batch,batch))
        while pending:
            ready,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                for r in future.result():
                    a,z,_=infer(model,r);j=positions[r[0]];assert not done[j];arrays[8][j]=a;arrays[32][j]=z;done[j]=True
                if done.sum()%256==0 or done.all():
                    for a in arrays.values():a.flush()
                    b.log(stage='official32' if official else 'replay32',games=int(done.sum()),total=len(ids),seconds=time.perf_counter()-start)
                assert time.perf_counter()-start<(3600 if official else 7200),'Bounded feature cap'
                batch=next(batches,None)
                if batch:pending.add(pool.submit(encode_batch,batch))
    assert done.all()
    for a in arrays.values():a.flush()
    b.write(dest/'RECEIPT.json',{'games':len(ids),'all_valid':True,'seconds':time.perf_counter()-start,'GPU':'local RTX3060','views':[8,32],'SHA256':{v:b.sha(dest/f'view{v}.npy') for v in (8,32)},'no_labels_or_new_fit_in_extraction':True})

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','reuse-B','extract','official-extract']);a=p.parse_args()
    with threadpool_limits(limits=4):{'prepare':prepare,'reuse-B':reuse_B,'extract':extract,'official-extract':lambda:extract(True)}[a.stage]()
if __name__=='__main__':main()
