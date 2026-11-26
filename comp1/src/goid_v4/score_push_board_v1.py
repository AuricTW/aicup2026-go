"""Packed real-game board/action pilot. No official test or reserved323 I/O."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import defaultdict
import hashlib,os,time,json
import numpy as np
from . import score_push_v1 as c

PACKED=(18*19*19+7)//8
VARIANTS=('state_action_order','state_only_order','action_only_order','state_action_mean')

def replay_batch(items):
    from .replay import replay_positions
    out=[]
    for index,text,expected,hd in items:
        assert hashlib.sha256(text.encode()).hexdigest()==expected and hd in (0,1)
        decisions=replay_positions(text,'B',include_legal_mask=False)
        assert decisions and len(decisions)<=16
        planes=np.zeros((16,18,19,19),np.uint8);scalars=np.zeros((16,14),np.float32);mask=np.zeros(16,np.bool_)
        passes=0
        for j,d in enumerate(decisions):
            assert d.target_color=='B' and d.state_planes.shape==(17,19,19)
            assert np.array_equal(d.history_mask,np.arange(8)<min(8,d.ply))
            p=d.p2_planes()
            assert p[-1].sum()==int(d.action!=361)
            if d.action!=361:assert d.board.flat[d.action]==0 and p[0].flat[d.action]==0 and p[1].flat[d.action]==0
            else:passes+=1
            planes[j]=p;scalars[j]=d.scalars();mask[j]=True
        packed=np.packbits(planes.reshape(16,-1),axis=-1)
        assert np.array_equal(np.unpackbits(packed,axis=-1)[:,:18*19*19],planes.reshape(16,-1))
        out.append((index,packed,scalars,mask,passes))
    return out

def export(indices,dest,*,workers=2):
    from .pilot_scope_v1 import PointReader
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False);indices=np.asarray(indices,np.int64)
    m=c.metadata();assert ((m['colors'][indices]==0)&(m['hd'][indices]<=1)).all()
    reserved=set(c.read(c.RUN/'reserved/EPISODES_METADATA_ONLY.json')['candidates'])-set(c.read(c.RUN/'EPISODES.json')['candidates'])
    assert not any(p.decode() in reserved for p in m['owners'][indices])
    reader=PointReader(c.ROOT/'data/processed/audit.sqlite');mapping={int(m['rows'][i]):int(i) for i in indices};texts={}
    try:
        for start in range(0,len(indices),800):
            batch=m['rows'][indices[start:start+800]]
            for row in reader.sample_text(batch):
                assert row['color']=='B';i=mapping[row['row_id']]
                texts[i]=(i,row['sgf_content'],m['raw'][i].decode(),int(m['hd'][i]))
        assert len(texts)==len(indices)
        query_plan=reader.plan;queries=reader.queries
    finally:reader.close()
    n=len(indices);packed=np.lib.format.open_memmap(dest/'planes_packed.npy',mode='w+',dtype=np.uint8,shape=(n,16,PACKED))
    scalars=np.lib.format.open_memmap(dest/'scalars.npy',mode='w+',dtype=np.float32,shape=(n,16,14))
    masks=np.lib.format.open_memmap(dest/'masks.npy',mode='w+',dtype=np.bool_,shape=(n,16))
    position={int(i):j for j,i in enumerate(indices)};batches=[list(texts.values())[i:i+64] for i in range(0,len(texts),64)]
    started=time.perf_counter();done=0;pass_count=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(replay_batch,batches,chunksize=1):
            for index,p,s,mask,passes in result:
                j=position[index];packed[j]=p;scalars[j]=s;masks[j]=mask;pass_count+=passes;done+=1
            if done%1024==0 or done==n:c.log(stage='H_real_replay',directory=dest.name,games=done,total=n,seconds=time.perf_counter()-started)
    packed.flush();scalars.flush();masks.flush();np.save(dest/'indices.npy',indices,allow_pickle=False)
    receipt={'games':n,'positions':int(masks.sum()),'passes_selected':pass_count,'seconds':time.perf_counter()-started,
        'packed_bytes':sum(p.stat().st_size for p in dest.iterdir() if p.is_file()),'workers':workers,
        'raw_hash_verified_per_allowed_game':True,'queries_by_exact_row_allowlist_only':queries,
        'target_pre_state_and_action_checks':True,'no_reserved323_games':True,'no_empty_or_failed_games_dropped':True,
        'files_sha256':{p.name:c.sha(p) for p in dest.iterdir() if p.is_file()},'code_sha256':c.sha(Path(__file__))}
    c.write(dest/'RECEIPT.json',receipt)
    return receipt

def prepare():
    dest=c.RUN/'H';dest.mkdir(exist_ok=False)
    c.write(dest/'ADMISSION_ADDENDUM.json',{
        'before_H_data_export_and_training':True,'pilot_updates':2000,'profile_checkpoint':500,'batch_P':4,'batch_K':2,
        'fit_inner_probe':'128 training identities;4 support training games and4 separate within-fit games each; not independent quality',
        'full_selection_inference_admission':{'finite':True,'effective_rank_min':4,'inner_top1_min':2/128,'positive_mean_true_minus_negative_cosine':True},
        'collapse_stop_after_profile':'effective_rank<2 AND mean_direction_norm>.99999',
        'pilot_max_wall_seconds':21600,'all_fits_exclude_union723':True,
        'mean_control':'no learned sequence or position encoding; per-decision pre-state/ply scalars remain, not fully order-free',
        'mixed_precision':False,'external_initialization':False,'GPU_memory_fraction_cap':.6,
        'code_sha256':c.sha(Path(__file__))})
    m=c.metadata();fit=np.load(c.RUN/'fitB_indices.npy');groups=defaultdict(list)
    for i in fit:groups[m['owners'][i].decode()].append(int(i))
    byrank=defaultdict(list)
    for p,ix in groups.items():
        if len(ix)>=28:byrank[str(m['ranks'][ix[0]])].append(p)
    players=[]
    while len(players)<128:
        for rank in sorted(byrank):
            available=sorted(set(byrank[rank])-set(players),key=lambda p:c.token('H_trainplayer_v1',p))
            if available:players.append(available[0])
            if len(players)==128:break
    records=[];members=[];inner=[]
    for p in players:
        ix=sorted(groups[p],key=lambda i:c.token('H_traingame_v1',m['group'][i].decode()))[:28]
        offset=len(records);records.extend(ix);members.append(list(range(offset,offset+24)));inner.append(list(range(offset+24,offset+28)))
    assert len(records)==3584 and len(set(records))==3584
    c.write(dest/'TRAIN_PLAN.json',{'members':members,'inner':inner,'players':players,'indices':records,
        'training_games':3072,'inner_games':512,'rank_counts':dict(__import__('collections').Counter(str(m['ranks'][groups[p][0]]) for p in players)),
        'full_selection_not_loaded_for_training':True,'no_reserved323_games':True})
    receipt=export(records,dest/'train_shards')
    c.log(stage='H_train_shards_complete',**{k:receipt[k] for k in ('games','positions','packed_bytes','seconds')})

def make_model(variant):
    import torch,math
    from torch import nn
    from torch.nn import functional as F
    from .board_models import BoardEncoder
    if variant not in VARIANTS:raise ValueError(variant)
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__();self.variant=variant
            self.board=BoardEncoder(18,channels=64,blocks=6)
            self.token=nn.Linear(64*2+14,128)
            layer=nn.TransformerEncoderLayer(128,4,256,.1,activation='gelu',batch_first=True)
            self.sequence=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.output=nn.Linear(128,256)
            if variant=='state_action_mean':self.sequence.requires_grad_(False)
        def forward(self,p,s,mask):
            if self.variant=='state_only_order':
                p=p.clone();p[:,:,-1]=0;s=s.clone();s[:,:,5]=0
            elif self.variant=='action_only_order':
                p=p.clone();p[:,:,:17]=0
            active=p[mask];h=self.board(active);action=active[:,-1:]
            tok=self.token(torch.cat((h.mean((-2,-1)),(h*action).sum((-2,-1)),s[mask]),dim=-1))
            full=torch.zeros((*mask.shape,128),dtype=tok.dtype,device=tok.device);full[mask]=tok
            if self.variant!='state_action_mean':
                pos=torch.arange(mask.shape[1],device=tok.device,dtype=tok.dtype)[:,None]
                freq=torch.exp(torch.arange(0,128,2,device=tok.device,dtype=tok.dtype)*(-math.log(10000)/128))
                pe=torch.zeros((mask.shape[1],128),device=tok.device,dtype=tok.dtype);pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
                full=self.sequence(full+pe[None],src_key_padding_mask=~mask)
            mean=full.masked_fill(~mask[:,:,None],0).sum(1)/mask.sum(1)[:,None]
            return F.normalize(self.output(mean),dim=-1,eps=1e-12)
    return Encoder()

def data(path):
    return tuple(np.load(Path(path)/name,mmap_mode='r',allow_pickle=False) for name in ('planes_packed.npy','scalars.npy','masks.npy'))

def batch(arrays,indices,device):
    import torch
    p,s,m=arrays
    planes=np.unpackbits(np.array(p[indices],copy=True),axis=-1)[...,:18*19*19].reshape(len(indices),16,18,19,19).astype(np.float32)
    return (torch.from_numpy(planes).to(device),torch.from_numpy(np.array(s[indices],copy=True)).to(device),torch.from_numpy(np.array(m[indices],copy=True)).to(device))

def encode(model,arrays,indices,device):
    import torch
    model.eval();result=[]
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        for start in range(0,len(indices),8):result.append(model(*batch(arrays,indices[start:start+8],device)).cpu().numpy())
    return np.concatenate(result)

def inner_probe(model,arrays,plan,device):
    from .score_push_metric_v1 import diagnostics
    support=[s[:4] for s in plan['members']];query=plan['inner']
    indices=[i for group in support+query for i in group]
    emb=encode(model,arrays,indices,device);sm=emb[:512].reshape(128,4,256).mean(1);qm=emb[512:].reshape(128,4,256).mean(1)
    sm/=np.maximum(np.linalg.norm(sm,axis=1,keepdims=True),1e-12);qm/=np.maximum(np.linalg.norm(qm,axis=1,keepdims=True),1e-12)
    scores=qm@sm.T;true=np.diag(scores);mean_neg=(scores.sum(1)-true)/127
    diag=diagnostics(emb);diag.update({'fit_inner_candidates':128,'support_games':4,'query_games':4,
        'top1':float((np.argmax(scores,axis=1)==np.arange(128)).mean()),'true_minus_mean_negative_cosine':float((true-mean_neg).mean()),
        'not_selection_or_confirmation_quality':True})
    return diag

def train(variant,seed=31415):
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from .pilot_train_v1 import runtime
    from .board_models import GE2ESoftmax
    device=runtime();torch.cuda.set_per_process_memory_fraction(.6);torch.manual_seed(seed)
    dest=c.RUN/'H'/f'{variant}_s{seed}';dest.mkdir(exist_ok=False)
    plan=c.read(c.RUN/'H/TRAIN_PLAN.json');arrays=data(c.RUN/'H/train_shards')
    model=make_model(variant).to(device);criterion=GE2ESoftmax().to(device)
    opt=torch.optim.AdamW([p for p in list(model.parameters())+list(criterion.parameters()) if p.requires_grad],lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();profile_time=None
    c.write(dest/'PREREG.json',{'variant':variant,'seed':seed,'updates':2000,'P':4,'K':2,'positions_max':16,
        'loss':'GE2E correct leave-one-game-out','fp32':True,'final_checkpoint_only':True,
        'code_sha256':c.sha(Path(__file__)),'protocol_sha256':c.sha(c.RUN/'PROTOCOL.json')})
    losses=[];seen=set()
    with sdpa_kernel(SDPBackend.MATH),(dest/'training.jsonl').open('x',encoding='utf-8') as f:
        for step in range(1,2001):
            if time.perf_counter()-start>21600:raise TimeoutError('H21600sec wall cap')
            ps=rng.choice(128,4,replace=False);ix=[int(i) for p in ps for i in rng.choice(plan['members'][p],2,replace=False)];seen.update(ix)
            model.train();emb=model(*batch(arrays,ix,device));loss=criterion(emb.reshape(4,2,256))
            opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(criterion.parameters()),5,error_if_nonfinite=True);opt.step()
            assert torch.isfinite(loss).item();losses.append(float(loss.detach()))
            if step==1 or step%100==0:
                event={'step':step,'loss_last100_mean':float(np.mean(losses[-100:])), 'seconds':time.perf_counter()-start,
                    'allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'gradient_norm':float(grad)}
                f.write(json.dumps(event)+'\n');f.flush();c.log(stage='H_training',variant=variant,**event)
            if step==500:
                profile_time=time.perf_counter()-start
                probe=inner_probe(model,arrays,plan,device)
                profile={'steps':500,'training_seconds':profile_time,'game_presentations':4000,'position_capacity_presentations':64000,
                    'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'position_capacity_per_second':64000/profile_time,
                    'profile_inference_probe':probe,'CPU_or_GPU_other_work_may_contend':True,'not_isolated_benchmark':True}
                c.write(dest/'PROFILE_500.json',profile)
                if probe['effective_rank']<2 and probe['mean_direction_norm']>.99999:
                    c.write(dest/'STOP.json',{'reason':'profile_collapse_gate','probe':probe,'no_selection_inference':True});return
        final_probe=inner_probe(model,arrays,plan,device)
    with (dest/'model.pt').open('xb') as f:torch.save({'encoder':{k:v.detach().cpu() for k,v in model.state_dict().items()},'GE2E':{k:v.detach().cpu() for k,v in criterion.state_dict().items()}},f)
    admission=final_probe['finite'] and final_probe['effective_rank']>=4 and final_probe['top1']>=2/128 and final_probe['true_minus_mean_negative_cosine']>0
    c.write(dest/'RESULT.json',{'status':'pilot_training_complete','variant':variant,'seed':seed,'steps':2000,'games_presented':16000,
        'unique_training_games':len(seen),'seconds_including_probes':time.perf_counter()-start,
        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'final_fit_inner_probe':final_probe,
        'admitted_to_full_selection_inference':bool(admission),'model_sha256':c.sha(dest/'model.pt'),
        'reserved323_quality_or_games_read':False})
    del model,criterion,opt;torch.cuda.empty_cache()

def run():
    prepare()
    train_all()

def train_all():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        for variant in VARIANTS:train(variant)
        full=c.RUN/'H/state_action_order_s31415/RESULT.json'
        if not full.exists() or not c.read(full)['admitted_to_full_selection_inference']:
            c.write(c.RUN/'H/DECISION.json',{'status':'full_model_failed_fit_inner_representation_admission','selection_not_inferred':True,
                'all_controls_preserved':True,'extra_seeds_or_larger_GPU_not_admitted':True,'reserved323_unopened':True});return
        selection()

def selection():
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from .pilot_train_v1 import runtime
    from .score_push_metric_v1 import cosine_scores
    device=runtime();torch.cuda.set_per_process_memory_fraction(.6)
    ep=c.read(c.RUN/'EPISODES.json');indices=np.asarray(sorted({i for s in ep['supports'] for i in s}|{i for q in ep['questions'] for i in q['indices']}),np.int64)
    shards=c.RUN/'H/selection_shards';export(indices,shards)
    arrays=data(shards);reference=np.load(c.RUN/'selection_prediction/RRF_D_PLDA_raw.npy')
    for variant in VARIANTS:
        directory=c.RUN/f'H/{variant}_s31415'
        if not (directory/'model.pt').exists():continue
        model=make_model(variant).to(device);model.load_state_dict(torch.load(directory/'model.pt',map_location='cpu',weights_only=True)['encoder'])
        started=time.perf_counter()
        with sdpa_kernel(SDPBackend.MATH):emb=encode(model,arrays,list(range(len(indices))),device)
        np.save(directory/'selection_embeddings.npy',emb,allow_pickle=False)
        score=cosine_scores(emb,indices,ep)
        c.evaluate_selection('H_'+variant,score)
        r=c.evaluate_selection('E_H_'+variant,c.fusion(reference,score))
        c.write(directory/'INFERENCE.json',{'games':len(indices),'seconds':time.perf_counter()-started,'reserved323_read':False,'selection_gate':r['passed_selection_gate']})
        del model;torch.cuda.empty_cache()
    # More seeds are gated on full state+action selection evidence; not a blind capacity escalation.
    first=c.read(c.RUN/'selection_results/E_H_state_action_order/SUMMARY.json')
    c.write(c.RUN/'H/DECISION.json',{'status':'all_admitted_control_comparisons_complete','first_seed_full_selection_gate':first['passed_selection_gate'],
        'requires_three_seed_check_before_confirmation':first['passed_selection_gate'],'reserved323_quality_read':False})

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','train_all','selection','run'],default='run')
    args=parser.parse_args()
    globals()[args.stage]()
