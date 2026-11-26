"""Train-only scalers/PLDA and matched few-shot heads; frozen promotion gates."""
from pathlib import Path
import sys,os,time,json,argparse,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
import v2_dense32_pilot_20261111_v1 as p
import numpy as np
from threadpoolctl import threadpool_limits
RUN=p.RUN;STEPS=2000

def assemble():
    rr=RUN/'remote_results';receipt=p.b.read(rr/'RECEIPT.json');assert receipt['all_valid']
    for f,h in receipt['SHA256'].items():assert p.b.sha(rr/f)==h
    ids=np.load(rr/'indices.npy');assert np.array_equal(ids,np.load(RUN/'replay_indices.npy'))
    cached=np.load(RUN/'cached_B_indices.npy');positions={int(i):('remote',j) for j,i in enumerate(ids)}
    positions.update({int(i):('B',j) for j,i in enumerate(cached)})
    dest=RUN/'views';dest.mkdir(exist_ok=False)
    for view in (8,32):
        source={'remote':np.load(rr/f'view{view}.npy',mmap_mode='r'),'B':np.load(RUN/f'cached_B/view{view}.npy',mmap_mode='r')}
        for scope in ('train','cal'):
            ii=np.load(RUN/f'{scope}_indices.npy');out=np.lib.format.open_memmap(dest/f'{scope}{view}.npy',mode='w+',dtype=np.float32,shape=(len(ii),2339))
            for j,i in enumerate(ii):where,k=positions[int(i)];out[j]=source[where][k]
            assert np.isfinite(out).all();out.flush()
    p.b.write(dest/'RECEIPT.json',{'train_games':16384,'cal_games':len(np.load(RUN/'cal_indices.npy')),'compression':'float16 tokens for both colors, summariesfloat32','cached_B_backend':'historical RTX3060','new_W_and_cal_backend':receipt['GPU'],'not_a_pure_hardware_causal_comparison':True,'Public_labels_loaded':False})

def pre(x,folder):
    with np.load(folder/'preprocess.npz',allow_pickle=False) as z:
        out=np.clip((np.asarray(x,np.float64)-z['mean'])/z['std'],-10,10).astype(np.float32);out[:,~z['active']]=0
    return out

def prepare_draws():
    rng=np.random.default_rng(p.SEED);ss=np.empty((STEPS,512,10),np.uint8);qq=np.zeros((STEPS,64,5),np.uint8);qp=np.empty((STEPS,64),np.uint16);nn=np.tile(np.asarray([1,2,3,5],np.uint8),16)
    for step in range(STEPS):
        for owner in range(512):ss[step,owner]=rng.choice(32,10,replace=False)
        qp[step]=rng.choice(512,64,replace=False)
        for j,owner in enumerate(qp[step]):
            remain=np.setdiff1d(np.arange(32),ss[step,owner]);qq[step,j,:nn[j]]=rng.choice(remain,int(nn[j]),replace=False)
    dest=RUN/'draws';dest.mkdir(exist_ok=False)
    for name,array in [('supports',ss),('queries',qq),('players',qp),('counts',nn)]:np.save(dest/f'{name}.npy',array,allow_pickle=False)
    p.b.write(dest/'RECEIPT.json',{'steps':STEPS,'seed':p.SEED,'support_query_no_shared_game':True,'same_draws_for_8_32':True,'SHA256':{f.name:p.b.sha(f) for f in dest.glob('*.npy')}})

def model_encode(model,x):
    import torch
    x=np.asarray(x,np.float32);x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);out=[]
    with torch.inference_mode():
        for start in range(0,len(x),128):
            real=x[start:start+128];value=np.pad(real,((0,128-len(real)),(0,0)));out.append(model(torch.from_numpy(value).to('cuda')).cpu().numpy()[:len(real)])
    return np.concatenate(out)

def train():
    import torch
    from goid.feature_metric import FeatureMetric
    p.b.runtime();labels=np.load(RUN/'train_labels.npy');assert np.array_equal(labels,np.repeat(np.arange(512),32))
    d=RUN/'models';d.mkdir(exist_ok=False)
    ss=np.load(RUN/'draws/supports.npy');qq=np.load(RUN/'draws/queries.npy');owners=np.load(RUN/'draws/players.npy');counts=np.load(RUN/'draws/counts.npy')
    rowbase=np.arange(512)[:,None]*32;mask=torch.from_numpy(np.arange(5)[None]<counts[:,None]).to('cuda');div=torch.from_numpy(counts.astype(np.float32)).to('cuda')[:,None]
    for view in (8,32):
        dest=d/f'view{view}';dest.mkdir();raw=np.load(RUN/f'views/train{view}.npy',mmap_mode='r');cal=np.load(RUN/f'views/cal{view}.npy',mmap_mode='r')
        mean=np.asarray(raw,np.float64).mean(0);std=np.asarray(raw,np.float64).std(0);active=std>1e-4;std=np.where(active,std,1.)
        np.savez(dest/'preprocess.npz',mean=mean,std=std,active=active);x=pre(raw,dest);xc=pre(cal,dest)
        fit=p.fit_plda(x,labels,dest/'raw_PLDA.npz',dimension=128);p.b.log(stage='raw_PLDA_fit',view=view,**fit)
        torch.manual_seed(p.SEED);torch.cuda.manual_seed_all(p.SEED)
        model=FeatureMetric(2339,128,512,.1).to('cuda');model.train();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
        normalized=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);tx=torch.from_numpy(normalized).to('cuda');history=[];start=time.perf_counter()
        for step in range(STEPS):
            si=(ss[step].astype(np.int64)+rowbase).ravel();qi=qq[step].astype(np.int64)+owners[step].astype(np.int64)[:,None]*32
            ids=torch.from_numpy(np.concatenate((si,qi.ravel()))).to('cuda');target=torch.from_numpy(owners[step].astype(np.int64)).to('cuda')
            opt.zero_grad(set_to_none=True);z=model(tx[ids]);support=torch.nn.functional.normalize(z[:5120].reshape(512,10,128).mean(1),dim=-1)
            query=z[5120:].reshape(64,5,128).masked_fill(~mask[:,:,None],0).sum(1)/div;query=torch.nn.functional.normalize(query,dim=-1)
            logits=(query@support.T)/.07;loss=torch.nn.functional.cross_entropy(logits,target);assert torch.isfinite(loss)
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(grad);opt.step()
            if step==0 or (step+1)%200==0:
                record={'step':step+1,'loss':float(loss.detach()),'accuracy_train':float((logits.argmax(1)==target).float().mean()),'seconds':time.perf_counter()-start};history.append(record);p.b.log(stage='episodic_10support',view=view,**record)
            assert time.perf_counter()-start<1200
        model.eval().requires_grad_(False);torch.save({'state_dict':model.cpu().state_dict(),'steps':STEPS,'seed':p.SEED,'view':view},dest/'episodic_head.pt');model.to('cuda')
        trainz=model_encode(model,x);calz=model_encode(model,xc);np.save(dest/'train_head.npy',trainz,allow_pickle=False);np.save(dest/'cal_head.npy',calz,allow_pickle=False)
        subset=np.arange(0,len(xc),max(1,len(xc)//8))[:8][::-1];sub=model_encode(model,xc[subset]);assert np.array_equal(calz[subset],sub)
        extra=p.fit_plda(trainz,labels,dest/'head_PLDA.npz',dimension=128)
        p.b.write(dest/'TRAIN_RECEIPT.json',{'steps':STEPS,'history':history,'seconds':time.perf_counter()-start,'raw_PLDA':fit,'head_PLDA':extra,'fixed128_head_subset_reverse_exact':True,'train_rows':len(x),'train_players':512,'Public_or_calibration_labels_in_fit':False,'model_SHA256':p.b.sha(dest/'episodic_head.pt')})
        del model,opt,tx;torch.cuda.empty_cache()
    p.b.write(d/'COMPLETION.json',{'complete':True,'same_draws':p.b.sha(RUN/'draws/RECEIPT.json'),'checkpoint_policy':'last, no early stopping on validation'})

def component_scores():
    calids=np.load(RUN/'cal_indices.npy');pos={int(i):j for j,i in enumerate(calids)}
    eps=[p.b.read(p.b.RUN/f'calibration/ep{i}.json') for i in range(2)]+[p.b.read(RUN/'EPISODE_CONFIRMATION.json')]
    dest=RUN/'component_predictions';dest.mkdir(exist_ok=False)
    for view in (8,32):
        folder=RUN/f'models/view{view}';raw=np.load(RUN/f'views/cal{view}.npy',mmap_mode='r')
        for label,file,features in [('raw','raw_PLDA.npz',pre(raw,folder)),('episodic','head_PLDA.npz',np.load(folder/'cal_head.npy'))]:
            model=p.PLDA(folder/file);z=model.transform(features)
            for j,ep in enumerate(eps):
                ss=[[pos[i] for i in s] for s in ep['supports']];qs=[[pos[i] for i in q['indices']] for q in ep['questions']]
                sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);v=model.score(qm,qn,sm,sn)
                take=np.arange(0,len(qs),37)[::-1];assert np.array_equal(v[take],model.score(qm[take],qn[take],sm,sn))
                pd=dest/f'ep{j}';pd.mkdir(exist_ok=True);np.save(pd/f'{label}{view}.npy',v,allow_pickle=False)
    p.b.write(dest/'RECEIPT.json',{'labels_not_used_in_scores':True,'fresh_episode_quality_not_evaluated':True,'subset_reverse_exact':True})

def values(score,ep):
    target=np.asarray([q['target'] for q in ep['questions']]);rank=p.b.ranks(score)[np.arange(len(target)),target]
    return np.where(rank<=5,np.exp(-(rank-1)),0.),rank

def compare(candidate,baseline,eps,minimum,net_min,short_floor=0.,long_floor=-.003):
    ps=sorted({q['player'] for ep in eps for q in ep['questions']});pos={v:i for i,v in enumerate(ps)};sums=np.zeros(len(ps));counts=np.zeros(len(ps));diff=[];net=0
    for ep,a,bb in zip(eps,candidate,baseline):
        av,ar=values(a,ep);bv,br=values(bb,ep);d=av-bv;diff.append(d);net+=int((ar==1).sum()-(br==1).sum())
        for q,v in zip(ep['questions'],d):i=pos[q['player']];sums[i]+=v;counts[i]+=1
    draws=np.random.default_rng(p.SEED).integers(len(ps),size=(5000,len(ps)));ci=np.quantile(sums[draws].sum(1)/counts[draws].sum(1),[.025,.975]).tolist()
    byq={q:float(np.concatenate([d[np.asarray([z['q']==q for z in ep['questions']])] for d,ep in zip(diff,eps)]).mean()) for q in (1,2,3,5)}
    mean=float(np.concatenate(diff).mean());checks={'mean':mean>=minimum,'each_episode':all(d.mean()>=0 for d in diff),'q1':byq[1]>=short_floor,'q2':byq[2]>=short_floor,'q3':byq[3]>=long_floor,'q5':byq[5]>=long_floor,'Top1':net>=net_min,'CI':ci[0]>0}
    return {'delta':mean,'episode_delta':[float(d.mean()) for d in diff],'by_games_delta':byq,'top1_net':net,'CI95':ci,'checks':checks,'passed':all(checks.values())}

def screen():
    eps=[p.b.read(p.b.RUN/f'calibration/ep{i}.json') for i in range(2)];baseline=[p.cp.candidate_scores(i)['F_C_H25'] for i in range(2)];results={};passing=[]
    for model in ('raw8','raw32','episodic8','episodic32'):
        comp=[np.load(RUN/f'component_predictions/ep{i}/{model}.npy') for i in range(2)]
        for weight in (.1,.25,.4):
            scores=[(1-weight)*s+weight/(60+p.b.ranks(k)) for s,k in zip(baseline,comp)];key=f'{model}_w{weight:g}';report=compare(scores,baseline,eps,.01,8)
            report.update({'model':model,'weight':weight,'metrics':[p.b.metrics(s,ep) for s,ep in zip(scores,eps)]});results[key]=report
            if report['passed']:passing.append(key)
    best=sorted(passing,key=lambda k:(-np.mean([z['score'] for z in results[k]['metrics']]),results[k]['weight'],0 if results[k]['model'].startswith('raw') else 1,int(results[k]['model'].replace('raw','').replace('episodic',''))))[0] if passing else None
    p.b.write(RUN/'SCREEN_RESULTS.json',results);p.b.write(RUN/'WINNER_FREEZE.json',{'winner':best,'passing':passing,'selected':results[best] if best else None,'fresh_confirmation_quality_not_yet_read':True,'Public_score_used_to_select':False})
    p.b.log(stage='dense32_screen',winner=best,passing=len(passing),summary={k:round(v['delta'],6) for k,v in results.items()})

def confirmation():
    selected=p.b.read(RUN/'WINNER_FREEZE.json');assert selected['winner'] is not None
    model=selected['selected']['model'];w=selected['selected']['weight'];ep=p.b.read(RUN/'EPISODE_CONFIRMATION.json')
    baseline=np.load(RUN/'fresh_baseline/F_C_H25.npy');k=np.load(RUN/f'component_predictions/ep2/{model}.npy');score=(1-w)*baseline+w/(60+p.b.ranks(k))
    result=compare([score],[baseline],[ep],.005,0,long_floor=-.005);result.update({'winner':selected['winner'],'metrics':p.b.metrics(score,ep),'baseline_metrics':p.b.metrics(baseline,ep),'same_players_fixed_support_not_independent':True})
    p.b.write(RUN/'CONFIRMATION_RESULT.json',result);p.b.log(stage='fresh_confirmation',**result)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['assemble','draws','train','predict','screen','confirm']);v=a.parse_args()
    with threadpool_limits(limits=4):{'assemble':assemble,'draws':prepare_draws,'train':train,'predict':component_scores,'screen':screen,'confirm':confirmation}[v.stage]()
