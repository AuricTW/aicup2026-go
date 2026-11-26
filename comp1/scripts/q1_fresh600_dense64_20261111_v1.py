"""Fresh600-player, nested32/64 frozen decision feature pilot, no uploads."""
from pathlib import Path
import sys,os,argparse,time,json,hashlib
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parent))
import v2_dense32_pilot_20261111_v1 as p
import train_v2_dense32_20261111_v2 as oldtrain
import numpy as np
from threadpoolctl import threadpool_limits
RUN=p.ROOT/'runs/q1_fresh600_dense64_20261111_v1';SEED=2026111171

def prepare():
    RUN.mkdir(exist_ok=False);m=p.b.meta();old600=set(p.b.read(p.b.RUN/'calibration/ep0.json')['candidates']);oldtrainids=np.load(p.ROOT/'runs/v2_q1_diversity875_20261111_v1/train_indices.npy');old875=set(m['owners'][oldtrainids].astype(str));histval=np.load(p.DENSE.parent/'validation_indices.npy');old723=set(m['owners'][histval].astype(str));banned=old600|old875|old723
    oldfit=np.load(p.RUN/'train_indices.npy');oldfamilies={f:set(m[f][oldfit]) for f in p.b.FAMILIES};eligible={};by=defaultdict(lambda:[[],[]])
    for i in np.flatnonzero((m['hd']<=1)&~np.isin(m['owners'].astype(str),sorted(banned))):
        if all(m[f][i] not in oldfamilies[f] for f in p.b.FAMILIES):by[m['owners'][i].decode()][int(int(m['group'][i][:8],16)%10>=7)].append(int(i))
    for owner,banks in by.items():
        unique=[]
        for bank in banks:
            used={f:set() for f in p.b.FAMILIES};a=[]
            for i in sorted(bank,key=lambda i:hashlib.sha256(m['group'][i]+str(SEED).encode()).digest()):
                if any(m[f][i] in used[f] for f in p.b.FAMILIES):continue
                a.append(i)
                for f in p.b.FAMILIES:used[f].add(m[f][i])
            unique.append(a)
        if len(unique[0])>=15 and len(unique[1])>=8:eligible[owner]=unique
    ranks=defaultdict(list)
    for owner,banks in eligible.items():ranks[str(m['ranks'][banks[0][0]])].append(owner)
    for rank in ranks:ranks[rank].sort(key=lambda owner:hashlib.sha256((str(SEED)+owner).encode()).digest())
    players=[];cursor=0
    while len(players)<600:
        changed=False
        for rank in sorted(ranks):
            if cursor<len(ranks[rank]):players.append(ranks[rank][cursor]);changed=True
            if len(players)==600:break
        assert changed,'not enough fresh600 candidates'
        cursor+=1
    players.sort();supports=[];used={f:set() for f in p.b.FAMILIES}
    for owner in players:
        s=[]
        for i in eligible[owner][0]:
            if any(m[f][i] in used[f] for f in p.b.FAMILIES):continue
            s.append(i)
            for f in p.b.FAMILIES:used[f].add(m[f][i])
            if len(s)==10:break
        assert len(s)==10
        supports.append(s)
    # Target identities are disjoint across screen/confirmation. All600 support
    # prototypes are public to the classifier, but no confirmation query labels
    # may be used in selection.
    targetorder=sorted(players,key=lambda owner:hashlib.sha256(('target|'+str(SEED)+owner).encode()).digest());eps=[]
    for part,owners,nq in [('screen',targetorder[:400],3),('confirmation',targetorder[400:],4)]:
        questions=[]
        for owner in owners:
            picked=[]
            for i in eligible[owner][1]:
                if any(m[f][i] in used[f] for f in p.b.FAMILIES):continue
                picked.append(i)
                for f in p.b.FAMILIES:used[f].add(m[f][i])
                if len(picked)==nq:break
            assert len(picked)==nq
            for i in picked:questions.append({'id':f'{part}_{len(questions):04}','indices':[i],'q':1,'target':players.index(owner),'player':owner})
        ep={'candidates':players,'supports':supports,'questions':questions,'seed':SEED,'part':part};eps.append(ep);p.b.write(RUN/f'EPISODE_{part}.json',ep)
    val=np.asarray(sorted({i for s in supports for i in s}|{i for ep in eps for z in ep['questions'] for i in z['indices']}),np.int64);assert len(val)==8000
    # New fits exclude the FULL owner/family scope of all600, not just chosen
    # games. Keep the historical exclusion723/old600 too.
    excluded=np.isin(m['owners'].astype(str),sorted(set(players)|old600|old723));safe=(~excluded)&(m['hd']<=1)
    for f in p.b.FAMILIES:safe &= ~np.isin(m[f],np.unique(m[f][excluded]))
    trainby=defaultdict(lambda:[[],[]]);oldB=np.load(p.DENSE.parent/'train_indices.npy')
    for i in oldB:
        if safe[i] and m['owners'][i].decode() in old875:trainby[m['owners'][i].decode()][0].append(int(i))
    for i in np.flatnonzero(safe&(m['colors']==1)):
        owner=m['owners'][i].decode()
        if owner in trainby:trainby[owner][1].append(int(i))
    trainplayers=sorted((owner for owner,banks in trainby.items() if min(map(len,banks))>=16),key=lambda owner:hashlib.sha256(('train|'+str(SEED)+owner).encode()).digest())[:512];assert len(trainplayers)==512
    train=[]
    for owner in trainplayers:
        for bank in trainby[owner]:train+=sorted(bank,key=lambda i:hashlib.sha256(m['group'][i]+str(SEED).encode()).digest())[:16]
    train=np.asarray(train,np.int64);labels=np.repeat(np.arange(512),32);B=train[m['colors'][train]==0];W=train[m['colors'][train]==1]
    for f in p.b.FAMILIES:assert not set(m[f][train])&set(m[f][val])
    assert not set(m['owners'][train])&set(m['owners'][val])
    for name,array in [('train_indices',train),('train_labels',labels),('cal_indices',val),('cached_B_indices',np.sort(B)),('replay_indices',np.asarray(sorted(set(W)|set(val)),np.int64))]:np.save(RUN/f'{name}.npy',array,allow_pickle=False)
    pass  # leaderboard record removed for the public release
    p.b.write(RUN/'PROTOCOL.json',{'created_utc':p.b.base.utc(),'source_SHA256':p.b.sha(__file__),'seed':SEED,'hypothesis':'reduce single-game decision-sampling noise by nested32vs64coverage; stop tuning old303identities',
      'train_players':512,'train_games':len(train),'training_colors':{'B':16,'W':16},'new600_candidates':600,'support':10,'screen':{'targets':400,'Q1_questions':1200},'confirmation':{'targets':200,'Q1_questions':800},'screen_confirmation_target_identity_overlap':0,
      'candidate_scope':'not old875neuraltrain/old600Nov11cal/historical723activevalidation; not globallypristine because historicalstatparents fit broadsafeoriginaldata',
      'all_new_fits_exclude_full600_identity_and4families_plus_old723_old600':True,'selectedval4families_disjoint_oldraw512fit':True,'original_sealed400_excluded_by_safe_source_not_opened':True,
      'features':'frozenHumanSL1d, HD0/1, actualB/W, nested16early16mid vs24early40mid, no>180; fixed32FP16 chunks pergame, float16tokens before2339summary',
      'original8_for_parent':'unquantizednative8summary fromnested4early4mid; parityprobe required; no newfit',
      'models':['raw128','raw256','joint256'],'views':[32,64],'joint':'concat normalizedHumanSL summary2339+sqrt histogram1582; allstandardization/trainbackground trainONLY',
      'fusion_weights':[.1,.25,.4],'fusion':'(1-w)*frozenraw32baseline+w/(60+ranknew), newQ1only; other1400rows exactraw32',
      'screen_64_vs_matched32_gate':{'score_delta_min':.01,'top1_net_min':6,'playercluster95_lower_gt':0},
      'screen_fusion_vs_raw32_gate':{'score_delta_min':.01,'top1_net_min':8,'playercluster95_lower_gt':0},
      'select':'highestmeanpassingfusion, tie smallerweight thenmodellexical; only64candidateseligible aftermatched32gate',
      'confirmation':'ONEfrozenwinner; allsamegates excepttop1netmin4 for800; requireboth64vs32andfusionbenefit, no retuning/extraquerybankif fail',
      'parent_validation_exposure':'legacyF/statMLP maysee theseidentities/games; fusedelta is secondary conditional not independentE2E. Newmetrics and matched32/64 trainingare player/family disjoint.',
      'compute':'4090 afterfreshnativeGPUowner/capacitycheck, bounded5400s perfeaturejob, CPUworkers <=8 ifidleCPUavailable; no drivervenvchanges',
      'official_features_only_after_gates':'onlynewQ1+support required, no officiallabels fitted','no_Public1_answers_no_sealed400_eval_no_upload':True})
    p.b.write(RUN/'SCOPE_AUDIT.json',{'eligible_fresh_players':len(eligible),'new600':600,'new_train512':512,'train_games':len(train),'replay_games':len(W)+len(val),'cached_B':len(B),'selected_games':len(val),'all_newfit_identity_and4family_overlap':0,'screen_target400_confirmation_target200_disjoint':True,'legacyparents_exposure_not_clean':True,'safe_metadata_SHA256':p.b.sha(p.b.SAFE)})
    files=[p.b.SAFE,p.RUN/'models/view32/preprocess.npz',p.RUN/'models/view32/raw_PLDA.npz',p.ROOT/'submissions/20261111_v2_dense32_raw32_w0.4_v1/submission_player.csv',p.ROOT/'submissions/20261111_v2_Q1_globaljoint075_challenger_v1/submission_player.csv']
    p.b.write(RUN/'PROTECTED_INPUTS_SHA256.json',{str(f):p.b.sha(f) for f in files});p.b.log(stage='fresh600_prepared',train=len(train),val=len(val),replay=len(W)+len(val))

def select(moves,target,canonical,quota=None):
    pools=[[],[]]
    for ply,(color,_) in enumerate(moves,1):
        if color==target.lower() and ply<=180:pools[int(ply>60)].append((hashlib.sha256(f'{canonical}|{target.lower()}|{ply}|20261104'.encode()).digest(),ply))
    quotas=(24,40) if quota is None else (quota,quota)
    return {ply for pool,n in zip(pools,quotas) for _,ply in sorted(pool)[:n]}

def encode(item):
    # Source-preserving copy generated for the larger nested selector, keeping
    # all board/rules/color/legality behavior identical to the original encoder.
    original=p.select
    try:p.select=select;return p.encode(item)
    finally:p.select=original

def infer64(model,record):
    import torch
    assert record[1] is not None and record[2] is None,(record[0],record[2]);px,glob,legal,actions,aux,meta,small,plies=record[1];count=len(actions);assert 1<=count<=64;zs=[];aa=[]
    for st in range(0,count,32):
        real=min(32,count-st);ix=np.pad(np.arange(st,st+real),(0,32-real),constant_values=st);device=next(model.parameters()).device
        x=np.unpackbits(px[ix],axis=1)[:,:22*361].reshape(32,22,19,19).astype(np.float32);mask=np.unpackbits(legal[ix],axis=1)[:,:362].astype(bool)
        pi,h=p.n.forward(model,torch.from_numpy(x).to(device),torch.from_numpy(glob[ix]).to(device),torch.from_numpy(np.repeat(meta[None],32,axis=0)).to(device),torch.from_numpy(mask).to(device),amp=True)
        h=h.float().flatten(2);ac=torch.from_numpy(actions[ix]).to(device);local=h.gather(2,ac.clamp_max(360)[:,None,None].expand(-1,384,1)).squeeze(2);local[ac==361]=0;expected=(h*pi[:,:361,None].transpose(1,2)).sum(2);z=torch.cat((local,h.mean(2),local-expected),1).cpu().numpy()[:real]
        chosen=pi.gather(1,ac[:,None]).squeeze(1);rank=((pi>chosen[:,None])&torch.from_numpy(mask).to(device)).sum(1).float()/torch.from_numpy(mask.sum(1)).to(device);entropy=-(pi*torch.log(pi.clamp_min(1e-12))).sum(1);top=pi.topk(2,dim=1).values
        numbers=torch.stack((torch.log(chosen.clamp_min(1e-12)),rank,entropy,torch.log(top[:,0].clamp_min(1e-12)),top[:,0]-top[:,1]),1).cpu().numpy()[:real];a=np.concatenate((aux[ix[:real],:2],numbers,aux[ix[:real],2:]),1);zs.append(z);aa.append(a)
    z=np.concatenate(zs);a=np.concatenate(aa);middle=np.zeros(count,bool)
    for phase,limit in ((0,16),(1,16)):
        at=np.flatnonzero((plies>60)==bool(phase));chosen=sorted(at,key=lambda j:hashlib.sha256(f'{record[0]}'.encode()).digest()) if False else None
        # Masks are generated by the exact selector and returned separately by
        # encode64_record; never infer nested hash order from chronology.
    middle=np.isin(plies,record[3]);original8=p.aggregate(z[small],a[small]);z=z.astype(np.float16).astype(np.float32)
    return original8,p.aggregate(z[middle],a[middle]),p.aggregate(z,a)

def cache():
    m=p.b.meta();ids=np.load(RUN/'cached_B_indices.npy');cacheids=np.load(p.DENSE/'indices.npy');pos={int(i):j for j,i in enumerate(cacheids)};tokens=np.load(p.DENSE/'tokens.npy',mmap_mode='r');aux=np.load(p.DENSE/'aux.npy',mmap_mode='r');plies=np.load(p.DENSE/'plies.npy',mmap_mode='r');counts=np.load(p.DENSE/'counts.npy');receipt=p.b.read(p.DENSE/'RECEIPT.json')
    for f in ('indices.npy','tokens.npy','aux.npy','plies.npy','counts.npy'):assert p.b.sha(p.DENSE/f)==receipt['files'][f]
    dest=RUN/'cached_B';dest.mkdir(exist_ok=False);arrays={v:np.empty((len(ids),2339),np.float32) for v in (32,64)}
    for j,i in enumerate(ids):
        k=pos[int(i)];pp=plies[k,:counts[k]];pools=[[],[]]
        for index,ply in enumerate(pp):pools[int(ply>60)].append((hashlib.sha256(f'{m["canonical"][i].decode()}|b|{int(ply)}|20261104'.encode()).digest(),index))
        for view,quotas in ((32,(16,16)),(64,(24,40))):
            which=sorted(index for pool,quota in zip(pools,quotas) for _,index in sorted(pool)[:quota]);arrays[view][j]=p.aggregate(tokens[k,which].astype(np.float32),aux[k,which])
    for view,a in arrays.items():np.save(dest/f'view{view}.npy',a,allow_pickle=False)
    p.b.write(dest/'RECEIPT.json',{'games':len(ids),'source_hashes_verified':True,'selected_trainB_only':True,'views':[32,64]});p.b.log(stage='fresh600_cached_B',games=len(ids))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','cache']);v=a.parse_args()
    with threadpool_limits(limits=4):{'prepare':prepare,'cache':cache}[v.stage]()
