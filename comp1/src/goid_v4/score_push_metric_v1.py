"""Controlled statistical-MLP budget/capacity pilot, union723 excluded."""
from pathlib import Path
import time
import numpy as np
import torch
from goid.feature_metric import FeatureMetric,PlayerBalancedSampler,supervised_contrastive_loss,FrozenFeaturePreprocessor
from . import score_push_v1 as c
from .pilot_train_v1 import runtime

def cosine_scores(emb,indices,ep):
    positions={int(v):i for i,v in enumerate(indices)}
    def profiles(sets):
        z=np.stack([emb[[positions[i] for i in s]].mean(0,dtype=np.float64) for s in sets])
        return np.array(z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12),np.float64,order='C')
    bank=profiles(ep['supports']);queries=profiles([q['indices'] for q in ep['questions']])
    out=np.empty((len(queries),len(bank)),np.float64)
    for i,q in enumerate(queries):out[i]=bank@np.array(q,order='C')
    return out

def encode(model,pre,raw,rows,device):
    model.eval();result=[]
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        for start in range(0,len(rows),1024):
            x=pre(raw[rows[start:start+1024]])
            if len(x)<1024:x=np.pad(x,((0,1024-len(x)),(0,0)))
            result.append(model(torch.from_numpy(x).to(device)).cpu().numpy()[:min(1024,len(rows)-start)])
    return np.concatenate(result)

def train(name,seed,*,hidden,dim,steps=20000):
    dest=c.RUN/'G'/f'{name}_s{seed}';dest.mkdir(parents=True,exist_ok=False)
    device=runtime();torch.cuda.set_per_process_memory_fraction(.6);torch.manual_seed(seed)
    m=c.metadata();ix=np.load(c.RUN/'fitB_indices.npy');ep=c.read(c.RUN/'EPISODES.json')
    raw=np.load(c.ROOT/'data/processed/features.npy',mmap_mode='r');pre=FrozenFeaturePreprocessor(c.RUN/'base_bundle','standardized',10)
    val=np.asarray(sorted({i for s in ep['supports'] for i in s}|{i for q in ep['questions'] for i in q['indices']}),np.int64)
    np.save(dest/'selection_indices.npy',val,allow_pickle=False)
    xnp=np.empty((len(ix),1582),np.float32)
    for start in range(0,len(ix),2048):xnp[start:start+2048]=pre(raw[m['rows'][ix[start:start+2048]]])
    x=torch.from_numpy(xnp).to(device);del xnp
    model=FeatureMetric(1582,dim,hidden,.1).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    sampler=PlayerBalancedSampler({j:m['owners'][i].decode() for j,i in enumerate(ix)},16,4,seed)
    torch.cuda.reset_peak_memory_stats();seen=np.zeros(len(ix),np.bool_);timer=time.perf_counter()
    c.write(dest/'PREREG.json',{'name':name,'seed':seed,'hidden':hidden,'embedding':dim,'steps':steps,
        'fitB_rows':len(ix),'fit_players':len(sampler.eligible),'controls_otherwise_fixed':True,'all_new_fit_excluded_players':723,
        'checkpoints':[5000,steps] if name=='G0' and seed==31415 else [steps],
        'code_sha256':c.sha(Path(__file__))})
    with (dest/'training.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,steps+1):
            if time.perf_counter()-timer>1800:raise TimeoutError('G per-run1800sec cap')
            model.train();ii,ll=sampler.sample();seen[ii]=True
            v=model(x[torch.from_numpy(ii).to(device)]);loss=supervised_contrastive_loss(v,torch.from_numpy(ll).to(device),.1)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
            if step==1 or step%5000==0:
                ev={'step':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-timer,'unique_games_presented':int(seen.sum()),'game_presentations':step*64}
                log.write(__import__('json').dumps(ev)+'\n');log.flush();c.log(stage='G_training',name=name,seed=seed,**ev)
            if (name=='G0' and seed==31415 and step==5000) or step==steps:
                checkpoint=dest/f'model_{step}.pt'
                with checkpoint.open('xb') as f:torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},f)
                emb=encode(model,pre,raw,m['rows'][val],device)
                np.save(dest/f'embeddings_{step}.npy',emb,allow_pickle=False)
                label=f'{name}_s{seed}_step{step}'
                score=cosine_scores(emb,val,ep)
                c.evaluate_selection(label,score)
                c.evaluate_selection('E_'+label,c.fusion(np.load(c.RUN/'selection_prediction/RRF_D_PLDA_raw.npy'),score))
    c.write(dest/'RESULT.json',{'status':'complete','steps':steps,'game_presentations':steps*64,'unique_games_presented':int(seen.sum()),
        'train_rows':len(ix),'train_players_eligible':len(sampler.eligible),'seconds_including_selection_inference':time.perf_counter()-timer,
        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'reserved323_quality_read':False,
        'collapse_diagnostics':diagnostics(emb),'checkpoint_sha256':c.sha(checkpoint)})
    del x,model,opt;torch.cuda.empty_cache()

def diagnostics(emb):
    sample=emb[:min(2048,len(emb))].astype(np.float64)
    s=np.linalg.svd(sample-sample.mean(0),compute_uv=False)
    p=s*s/max(float(np.sum(s*s)),1e-12)
    return {'mean_direction_norm':float(np.linalg.norm(sample.mean(0))),
        'effective_rank':float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))),
        'norm_mean':float(np.linalg.norm(sample,axis=1).mean()),'finite':bool(np.isfinite(emb).all())}

def run():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        train('G0',31415,hidden=512,dim=128)
        train('G1',31415,hidden=1024,dim=256)
        choices=[]
        for name in ('G0','G1'):
            key=f'E_{name}_s31415_step20000';r=c.read(c.RUN/f'selection_results/{key}/SUMMARY.json')
            if r['passed_selection_gate']:choices.append((r['metrics']['short'],name))
        if not choices:
            c.write(c.RUN/'G/DECISION.json',{'status':'no_first_seed_passes_common_selection_gate','extra_seeds_not_admitted':True,'G2_not_admitted':True,'reserved323_quality_read':False})
            return
        _,name=max(choices);hidden,dim=(512,128) if name=='G0' else (1024,256)
        for seed in c.SEEDS[1:]:train(name,seed,hidden=hidden,dim=dim)
        ep=c.read(c.RUN/'EPISODES.json');indices=np.load(c.RUN/f'G/{name}_s31415/selection_indices.npy')
        emb=np.concatenate([np.load(c.RUN/f'G/{name}_s{s}/embeddings_20000.npy')/np.sqrt(3) for s in c.SEEDS],axis=1)
        np.save(c.RUN/f'G/{name}_three_seed_embeddings.npy',emb,allow_pickle=False)
        score=cosine_scores(emb,indices,ep);c.evaluate_selection(f'{name}_three_seed',score)
        result=c.evaluate_selection(f'E_{name}_three_seed',c.fusion(np.load(c.RUN/'selection_prediction/RRF_D_PLDA_raw.npy'),score))
        c.write(c.RUN/'G/DECISION.json',{'status':'selected_first_seed_branch_then_three_seed_check','branch':name,
            'three_seed_gate_passed':result['passed_selection_gate'],'G2_not_admitted_this_bounded_round':True,'reserved323_quality_read':False})

if __name__=='__main__':run()
