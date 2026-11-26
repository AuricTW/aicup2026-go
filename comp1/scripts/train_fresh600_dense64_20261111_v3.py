"""Train-only paired32/64 metrics and target-disjoint screen/confirmation."""
from pathlib import Path
import sys,argparse,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import q1_fresh600_dense64_20261111_v1 as q
import clean_proxy_fresh600_dense64_20261111_v2 as clean
import v2_color_plda_20261111_v1 as cp
import numpy as np
from threadpoolctl import threadpool_limits
p=q.p;RUN=q.RUN;MODELS=(('raw128',128),('raw256',256),('joint256',256))

def baseline_stat():
    clean.build();clean.stat_predictions()

def assemble():
    rr=RUN/'remote_results';receipt=p.b.read(rr/'RECEIPT.json');assert receipt['all_valid']
    for f,h in receipt['SHA256'].items():assert p.b.sha(rr/f)==h
    rids=np.load(rr/'indices.npy');assert np.array_equal(rids,np.load(RUN/'replay_indices.npy'));cached=np.load(RUN/'cached_B_indices.npy');where={int(i):('remote',j) for j,i in enumerate(rids)};where.update({int(i):('B',j) for j,i in enumerate(cached)})
    out=RUN/'views';out.mkdir(exist_ok=False)
    for view in (32,64):
        sources={'remote':np.load(rr/f'view{view}.npy',mmap_mode='r'),'B':np.load(RUN/f'cached_B/view{view}.npy',mmap_mode='r')}
        for role in ('train','cal'):
            ids=np.load(RUN/f'{role}_indices.npy');a=np.stack([sources[where[int(i)][0]][where[int(i)][1]] for i in ids]);assert np.isfinite(a).all();np.save(out/f'{role}{view}.npy',a,allow_pickle=False)
    # Parent frozen8 needs original unquantized source; all validation games
    # are replayed, no B-only historical token cache in this path.
    cal=np.load(RUN/'cal_indices.npy');lookup={int(i):j for j,i in enumerate(rids)};a=np.load(rr/'view8.npy',mmap_mode='r');np.save(out/'cal_original8.npy',np.stack([a[lookup[int(i)]] for i in cal]),allow_pickle=False)
    p.b.write(out/'RECEIPT.json',{'roles_assembled':True,'train512':16384,'validation600':8000,'cached_B_3060_remote_other4090':True,'no_quality':True})

def hist(ids):
    m=p.b.meta();raw=np.asarray(np.load(p.ROOT/'data/processed/features.npy',mmap_mode='r')[m['rows'][ids]],np.float32);return cp.histograms(raw,p.b.read(p.ROOT/'data/processed/feature_schema.json'),sqrt=True)

def prepare_x(raw,folder,h=None,fit=False):
    if fit:
        mean=np.asarray(raw,np.float64).mean(0);std=np.asarray(raw,np.float64).std(0);active=std>1e-4;np.savez(folder/'preprocess.npz',mean=mean,std=np.where(active,std,1.),active=active)
    x=q.oldtrain.pre(raw,folder)
    if h is not None:
        if fit:
            mean=np.asarray(h,np.float64).mean(0);std=np.asarray(h,np.float64).std(0);active=std>1e-4;np.savez(folder/'hist_preprocess.npz',mean=mean,std=np.where(active,std,1.),active=active)
        with np.load(folder/'hist_preprocess.npz') as z:y=np.clip((h.astype(np.float64)-z['mean'])/z['std'],-10,10).astype(np.float32);y[:,~z['active']]=0
        x=np.concatenate((x,y),axis=1)
    return x

def train():
    dest=RUN/'models';dest.mkdir(exist_ok=False);ids=np.load(RUN/'train_indices.npy');labels=np.load(RUN/'train_labels.npy');hh=hist(ids)
    for view in (32,64):
        raw=np.load(RUN/f'views/train{view}.npy',mmap_mode='r')
        for name,dim in MODELS:
            folder=dest/f'{name}_v{view}';folder.mkdir();x=prepare_x(raw,folder,hh if name.startswith('joint') else None,fit=True);fit=p.fit_plda(x,labels,folder/'PLDA.npz',dimension=dim);p.b.write(folder/'FIT_RECEIPT.json',fit);p.b.log(stage='fresh600_metricfit',view=view,model=name,**fit)

def component(folder,raw,ids,eps,hh=None):
    order=np.argsort(ids,kind='stable');ids=ids[order];raw=raw[order];hh=None if hh is None else hh[order];x=prepare_x(raw,folder,hh if folder.name.startswith('joint') else None);model=p.PLDA(folder/'PLDA.npz');z=model.transform(x);lookup={int(i):j for j,i in enumerate(ids)};out=[]
    for ep in eps:
        ss=[[lookup[i] for i in s] for s in ep['supports']];qs=[[lookup[i] for i in v['indices']] for v in ep['questions']];sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);value=model.score(qm,qn,sm,sn);take=np.arange(0,len(qs),53)[::-1];assert np.array_equal(value[take],model.score(qm[take],qn[take],sm,sn));out.append(value)
    return out

def parent():
    clean.parent()

def values(score,ep):
    targets=np.asarray([z['target'] for z in ep['questions']]);r=p.b.ranks(score)[np.arange(len(targets)),targets];return np.where(r<=5,np.exp(-(r-1)),0.),r

def compare(a,b,ep,minmean=.01,minnet=6):
    av,ar=values(a,ep);bv,br=values(b,ep);delta=av-bv;owners=sorted({z['player'] for z in ep['questions']});pos={v:i for i,v in enumerate(owners)};sums=np.zeros(len(owners));count=np.zeros(len(owners))
    for z,v in zip(ep['questions'],delta):i=pos[z['player']];sums[i]+=v;count[i]+=1
    draws=np.random.default_rng(q.SEED).integers(len(owners),size=(5000,len(owners)));ci=np.quantile(sums[draws].sum(1)/count[draws].sum(1),[.025,.975]).tolist();net=int((ar==1).sum()-(br==1).sum());checks={'mean':float(delta.mean())>=minmean,'Top1':net>=minnet,'CI':ci[0]>0}
    return {'delta':float(delta.mean()),'CI95':ci,'Top1_net':net,'score':float(av.mean()),'baseline_score':float(bv.mean()),'Top1':int((ar==1).sum()),'baseline_Top1':int((br==1).sum()),'Top5':int((ar<=5).sum()),'baseline_Top5':int((br<=5).sum()),'n':len(av),'players':len(owners),'checks':checks,'passed':all(checks.values())}

def fuse(base,comp,part,route,w):
    if route=='add':return (1-w)*base+w/(60+p.b.ranks(comp))
    old=np.load(RUN/f'parent/raw32_{part}.npy')
    return base+.4*w*(1/(60+p.b.ranks(comp))-1/(60+p.b.ranks(old)))

def screen():
    dest=RUN/'screen';dest.mkdir(exist_ok=False);ids=np.load(RUN/'cal_indices.npy');ep=p.b.read(RUN/'EPISODE_screen.json');hh=hist(ids);base=np.load(RUN/'parent/screen.npy');components={};paired={};results={};passing=[]
    for name,_ in MODELS:
        for view in (32,64):
            value=component(RUN/f'models/{name}_v{view}',np.load(RUN/f'views/cal{view}.npy'),ids,[ep],hh)[0];components[f'{name}_v{view}']=value;np.save(dest/f'{name}_v{view}.npy',value,allow_pickle=False)
        paired[name]=compare(components[f'{name}_v64'],components[f'{name}_v32'],ep)
        for route,weights in [('add',(.1,.25,.4)),('replace',(.25,.5,1.))]:
            for w in weights:
                score=fuse(base,components[f'{name}_v64'],'screen',route,w);report=compare(score,base,ep,minnet=8);report.update({'model':name,'weight':w,'route':route,'paired64_vs32_gate':paired[name]['passed']});report['eligible']=report['passed'] and paired[name]['passed'];key=f'{name}_v64_{route}_w{w:g}';results[key]=report
                if report['eligible']:passing.append(key)
    best=sorted(passing,key=lambda k:(-results[k]['score'],0 if results[k]['route']=='replace' else 1,results[k]['weight'],results[k]['model']))[0] if passing else None
    p.b.write(RUN/'SCREEN_RESULTS.json',{'paired64_vs32':paired,'fusion':results});p.b.write(RUN/'WINNER_FREEZE.json',{'winner':best,'selected':results[best] if best else None,'passing':passing,'confirmation_target_outcomes_unread':True,'clean_proxy_not_exact_deployed_raw32':True});p.b.log(stage='fresh600_screen',winner=best,paired=paired,results=results)

def confirm():
    f=p.b.read(RUN/'WINNER_FREEZE.json');assert f['winner'];sel=f['selected'];ep=p.b.read(RUN/'EPISODE_confirmation.json');ids=np.load(RUN/'cal_indices.npy');hh=hist(ids);comp={}
    for view in (32,64):comp[view]=component(RUN/f'models/{sel["model"]}_v{view}',np.load(RUN/f'views/cal{view}.npy'),ids,[ep],hh)[0]
    paired=compare(comp[64],comp[32],ep,minnet=4);base=np.load(RUN/'parent/confirmation.npy');score=fuse(base,comp[64],'confirmation',sel['route'],sel['weight']);fusion=compare(score,base,ep,minnet=4);result={'passed':paired['passed'] and fusion['passed'],'winner':f['winner'],'paired64_vs32':paired,'fusion':fusion,'confirmation_targets200_disjoint_screen400':True,'newmetrics_identity_and4family_disjoint':True,'clean_proxy_not_exact_deployed_raw32':True,'route':sel['route']}
    np.save(RUN/'confirmation_fused.npy',score,allow_pickle=False);np.save(RUN/'confirmation_component.npy',comp[64],allow_pickle=False);p.b.write(RUN/'CONFIRMATION_RESULT.json',result);p.b.log(stage='fresh600_confirmation',**result)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['baseline-stat','assemble','train','parent','screen','confirm']);v=a.parse_args()
    with threadpool_limits(limits=4):{'baseline-stat':baseline_stat,'assemble':assemble,'train':train,'parent':parent,'screen':screen,'confirm':confirm}[v.stage]()
