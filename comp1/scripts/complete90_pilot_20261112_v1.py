"""All target decisions through ply180, paired64/90, bounded train512x16 pilot."""
from pathlib import Path
import sys,argparse,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
import heterovariance64_20261112_v1 as rejected
cap=rejected.cap;p=cap.p;t=cap.t;q=cap.q;PRIOR=cap.PRIOR
import numpy as np
from threadpoolctl import threadpool_limits
RUN=p.ROOT/'runs/complete90_pilot_20261112_v1';SEED=2026111221
PARENT=rejected.PARENT


def prepare():
    RUN.mkdir(exist_ok=False);assert p.b.read(rejected.RUN/'WINNER_FREEZE.json')['winner'] is None and not (rejected.RUN/'CONFIRMATION_RESULT.json').exists();m=p.b.meta();ep=p.b.read(rejected.RUN/'EPISODE_screen.json');conf=p.b.read(rejected.RUN/'EPISODE_confirmation.json');p.b.write(RUN/'EPISODE_screen.json',ep);p.b.write(RUN/'EPISODE_confirmation.json',conf)
    oldtrain=np.load(PRIOR/'train_indices.npy');train=[]
    for j in range(512):
        block=oldtrain[j*32:(j+1)*32];assert len(set(m['owners'][block]))==1
        for c in (0,1):train+=sorted(block[m['colors'][block]==c].tolist(),key=lambda i:hashlib.sha256(m['group'][i]+str(SEED).encode()).digest())[:8]
    train=np.asarray(train,np.int64);labels=np.repeat(np.arange(512),16);assert len(train)==8192 and len(set(train))==8192
    screenids=np.asarray(sorted({i for s in ep['supports'] for i in s}|{i for z in ep['questions'] for i in z['indices']}),np.int64);confirmids=np.load(rejected.RUN/'cal_indices.npy');ids=np.asarray(sorted(set(train)|set(screenids)|set(confirmids)),np.int64)
    for f in p.b.FAMILIES:assert not set(m[f][train])&set(m[f][np.union1d(screenids,confirmids)])
    assert not set(m['owners'][train])&set(m['owners'][np.union1d(screenids,confirmids)])
    for name,a in [('train_indices',train),('train_labels',labels),('screen_indices',screenids),('cal_indices',confirmids),('replay_indices',ids)]:np.save(RUN/f'{name}.npy',a,allow_pickle=False)
    protected=p.b.read(rejected.RUN/'PROTECTED_INPUTS_SHA256.json');protected[str(rejected.RUN/'EPISODE_confirmation.json')]=p.b.sha(rejected.RUN/'EPISODE_confirmation.json');protected[str(cap.__file__)]=p.b.sha(cap.__file__)
    for file,h in protected.items():assert p.b.sha(file)==h,file
    p.b.write(RUN/'PROTECTED_INPUTS_SHA256.json',protected)
    p.b.write(RUN/'PROTOCOL.json',{'created_utc':p.b.base.utc(),'source_SHA256':p.b.sha(__file__),'metadata_seed':SEED,'bootstrap_seed':cap.SEED,'hypothesis':'completeall targetcolor decisions inply1..180 reduces64point samplingnoise; early30+middle60 maximum90; no latergame newcontext','features':'same frozenHumanSL rank1d,targetcolorHD0/1,summary2339; nested8unquantized, paired64quantized vs90quantized; pergame32FP16 chunks','train':'bounded512playersx16games =8192,8B+8W, stablemetadataonly subsetofprior512x32; all600owner/familyexcluded','controls':'same8192games,higher/lowercoverage on sameGPUforward; separatetrain-onlyscalers andPLDA256/768 each64/90','new_models':['raw256_v64','raw768_v64','raw256_v90','raw768_v90'],'component_modes':['raw256','raw768','mean'],'alphas':[.25,.5,1.],'routes':['Q1','all'],'configs':18,'baseline':'currentfrozen cleanproxyprior plus0.4Q1/0.7Q235 equalRRF256/768 fromfull512x32; deployedofficialbest retained','fusion':'replacealpha ofcurrentneuralensemble withselected90model(s)RRF60; priorandfinalweightsfixed','screen':p.b.read(cap.RUN/'SCOPE_AUDIT.json')['screen_by_games'],'confirmation':'sameunused metadata-frozen bankfromrejectedhetero branch, neverfeaturizedorevaluated;200targetdisjointscreen,1000newgames,800nestedqueries; sameexposed600identitypool NOT independentidentityholdout','gates_both_stages':p.b.read(cap.RUN/'PROTOCOL.json')['gates_both_stages'],'additional_gate':'selected90component routedmacroSID >=matched64component on same8192fit, CI reported without requiringpositivepairedCI; fusionCI gate stillrequired','selection':'highestpassing4lengthmacroSID;tie fewerroute,smaller alpha,modelexical; ONEwinnerconfirmation,no runnerups or extra bankafterfailedconfirmation','if_fail':'nofailedcandidate export; keepallnegativeevidence','fallback_encoding':'ifcomplete90encoding invalid butoriginal64valid,use64forbothviews,keepcaseandlogreason; technicalstopif>1% fallback; no quality filtering','engineering_before_quality':['nested64inputs exactvsoriginal64','native64vsnested64 maxabs<0.005 andnormalizedRMS<0.005','native8parity samebounds','coldwarmreverseprobe exact','all90selecteddecisions <=180 andtargetcolor'],'fitting':'CPU4threads, samePLDA withinshrink0.1; no neuralfinetune','GPU':'4090onlyifnativeidle, perrunownedPID,cap5400s train/devfeaturejob andofficialjob each, no environmentchange','Public_answer_key_not_loaded':True,'original_sealed400_not_opened':True,'no_upload':True})
    p.b.write(RUN/'SCOPE_AUDIT.json',{'train_players':512,'train_games':8192,'B_perplayer':8,'W_perplayer':8,'screen_uniquegames':len(screenids),'confirm_uniquegames':len(confirmids),'total_replay_uniquegames':len(ids),'screen300_confirm200_targetdisjoint':True,'original512x32_models_preserved':True,'all_fit_owner_andfamily_overlap':0,'confirmation_quality_unread':True,'rejectedbranch_quality_didnot_open_newbank':True,'safe_metadata_SHA256':p.b.sha(p.b.SAFE)});p.b.log(stage='complete90_prepared',train=8192,total_replay=len(ids))


def pre(raw,folder,fit=False):
    if fit:
        mean=np.asarray(raw,np.float64).mean(0);std=np.asarray(raw,np.float64).std(0);active=std>1e-4;np.savez(folder/'preprocess.npz',mean=mean,std=np.where(active,std,1),active=active)
    return q.oldtrain.pre(raw,folder)


def train():
    receipt=p.b.read(RUN/'dev_results/RECEIPT.json');assert receipt['all_valid'];ids=np.load(RUN/'dev_results/indices.npy');pos={int(i):j for j,i in enumerate(ids)};train=np.load(RUN/'train_indices.npy');at=[pos[int(i)] for i in train];labels=np.load(RUN/'train_labels.npy');dest=RUN/'models';dest.mkdir(exist_ok=False)
    for view in (64,90):
        raw=np.load(RUN/f'dev_results/view{view}.npy',mmap_mode='r')[at]
        for dim in (256,768):
            folder=dest/f'raw{dim}_v{view}';folder.mkdir();x=pre(raw,folder,True);r=p.fit_plda(x,labels,folder/'PLDA.npz',dimension=dim);p.b.write(folder/'FIT_RECEIPT.json',dict(r,view=view,training_only=True));p.b.log(stage='complete90_fit',view=view,**r)
    p.b.write(RUN/'MODELS_SHA256.json',{str(f):p.b.sha(f) for f in dest.rglob('*') if f.is_file()})


def component(folder,raw,ids,ep):
    order=np.argsort(ids,kind='stable');ids=ids[order];raw=raw[order];model=p.PLDA(folder/'PLDA.npz');z=model.transform(pre(raw,folder));pos={int(i):j for j,i in enumerate(ids)};ss=[[pos[i] for i in s] for s in ep['supports']];qs=[[pos[i] for i in z['indices']] for z in ep['questions']];sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);score=model.score(qm,qn,sm,sn);take=np.arange(0,len(qs),37)[::-1];assert np.array_equal(score[take],model.score(qm[take],qn[take],sm,sn));return score


def mixture(c,mode,view):
    if mode=='mean':return .5/(60+p.b.ranks(c[f'raw256_v{view}']))+.5/(60+p.b.ranks(c[f'raw768_v{view}']))
    return 1/(60+p.b.ranks(c[f'{mode}_v{view}']))


def changed(base,old,new,ep,alpha,route):
    out=base.copy();mask=np.asarray([route=='all' or z['q']==1 for z in ep['questions']]);cand=base+cap.countweights(ep)*alpha*(new-old);out[mask]=cand[mask];return out


def screen():
    ep=p.b.read(RUN/'EPISODE_screen.json');ids=np.load(RUN/'dev_results/indices.npy');components={}
    for view in (64,90):
        raw=np.load(RUN/f'dev_results/view{view}.npy',mmap_mode='r')
        for dim in (256,768):name=f'raw{dim}_v{view}';components[name]=component(RUN/'models'/name,raw,ids,ep)
    old=.5/(60+p.b.ranks(np.load(cap.RUN/'screen/global64.npy')))+.5/(60+p.b.ranks(np.load(cap.RUN/'screen/raw768.npy')));prior=np.load(cap.RUN/'screen/prior.npy');w=cap.countweights(ep);base=(1-w)*prior+w*old;dest=RUN/'screen';dest.mkdir(exist_ok=False)
    for name,a in [('baseline',base),('oldneural',old)]+list(components.items()):np.save(dest/f'{name}.npy',a,allow_pickle=False)
    results={};passing=[]
    for mode in ('raw256','raw768','mean'):
        new=mixture(components,mode,90);control=mixture(components,mode,64)
        for alpha in (.25,.5,1.):
            for route in ('Q1','all'):
                score=changed(base,old,new,ep,alpha,route);r=cap.compare(score,base,ep,route);paired=cap.compare(new,control,ep,route);r.update({'mode':mode,'alpha':alpha,'route':route,'paired90vs64':paired});r['checks']['paired_nonnegative']=paired['delta']>=0;r['passed']=all(r['checks'].values());key=f'{mode}_a{alpha:g}_{route}';results[key]=r
                if r['passed']:passing.append(key)
    best=sorted(passing,key=lambda k:(-results[k]['macro_score_all'],1 if results[k]['route']=='Q1' else 4,results[k]['alpha'],results[k]['mode']))[0] if passing else None;p.b.write(RUN/'SCREEN_RESULTS.json',results);p.b.write(RUN/'WINNER_FREEZE.json',{'winner':best,'selected':results[best] if best else None,'passing':passing,'confirmation_quality_unread':True});p.b.log(stage='complete90_screen',winner=best,passing=passing,summary={k:{'delta':r['delta'],'byq':r['by_games_delta'],'Top1net':r['Top1_net'],'paired_delta':r['paired90vs64']['delta']} for k,r in results.items()})


def confirm():
    sel=p.b.read(RUN/'WINNER_FREEZE.json')['selected'];assert sel;ep=p.b.read(RUN/'EPISODE_confirmation.json');ids=np.load(RUN/'dev_results/indices.npy');components={}
    dims=(256,768) if sel['mode']=='mean' else (int(sel['mode'][3:]),)
    for view in (64,90):
        raw=np.load(RUN/f'dev_results/view{view}.npy',mmap_mode='r')
        for dim in dims:name=f'raw{dim}_v{view}';components[name]=component(RUN/'models'/name,raw,ids,ep)
    # Fresh baseline uses original frozen full512x32 metrics and samefresh64.
    wanted=np.load(RUN/'cal_indices.npy');pos={int(i):j for j,i in enumerate(ids)};at=[pos[int(i)] for i in wanted];raw64=np.load(RUN/'dev_results/view64.npy',mmap_mode='r')[at];raw8=np.load(RUN/'dev_results/view8.npy',mmap_mode='r')[at];m=p.b.meta();local={int(i):j for j,i in enumerate(wanted)};ss=[[local[i] for i in s] for s in ep['supports']];qs=[[local[i] for i in z['indices']] for z in ep['questions']];features=np.asarray(np.load(p.ROOT/'data/processed/features.npy',mmap_mode='r')[m['rows'][wanted]],np.float32);hist=t.cp.histograms(features,p.b.read(p.ROOT/'data/processed/feature_schema.json'),sqrt=True);codes=p.b.sequence_export(wanted,m);gm=p.PLDA(PRIOR/'clean_proxy/models/global_PLDA.npz');z=gm.transform(hist);sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);gs=gm.score(qm,qn,sm,sn);cs=t.cp.color_score(features,m['colors'][wanted],ss,qs,PRIOR/'clean_proxy/models');seq=t.clean.sequence_scores(codes,m['colors'][wanted],ss,qs);fc=t.cp.fixed_rrf60(t.cp.fixed_rrf60(gs,cs),seq);hm=p.PLDA(PRIOR/'clean_proxy/models/H8/PLDA.npz');z=hm.transform(q.oldtrain.pre(raw8,PRIOR/'clean_proxy/models/H8'));sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);hs=hm.score(qm,qn,sm,sn);prior=.75*fc+.25/(60+p.b.ranks(hs));g=t.component(PRIOR/'models/raw256_v64',raw64,wanted,[ep])[0];g768=cap.component(cap.RUN/'models/raw768',raw64,wanted,ep);old=.5/(60+p.b.ranks(g))+.5/(60+p.b.ranks(g768));w=cap.countweights(ep);base=(1-w)*prior+w*old;new=mixture(components,sel['mode'],90);control=mixture(components,sel['mode'],64);score=changed(base,old,new,ep,sel['alpha'],sel['route']);r=cap.compare(score,base,ep,sel['route']);paired=cap.compare(new,control,ep,sel['route']);r['checks']['paired_nonnegative']=paired['delta']>=0;r['passed']=all(r['checks'].values());r.update({'winner':p.b.read(RUN/'WINNER_FREEZE.json')['winner'],'mode':sel['mode'],'alpha':sel['alpha'],'route':sel['route'],'paired90vs64':paired,'sameexposed600_notindependent':True,'fresh1000games_namedbanks_disjoint':True,'screen300_confirm200_targetdisjoint':True,'clean_proxy_notexactproduction':True});dest=RUN/'confirmation';dest.mkdir(exist_ok=False)
    for name,a in [('baseline',base),('candidate',score),('oldneural',old),('newneural',new)]+list(components.items()):np.save(dest/f'{name}.npy',a,allow_pickle=False)
    p.b.write(RUN/'CONFIRMATION_RESULT.json',r);p.b.log(stage='complete90_confirmation',**r)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','screen','confirm']);v=parser.parse_args()
    with threadpool_limits(limits=4):{'prepare':prepare,'train':train,'screen':screen,'confirm':confirm}[v.stage]()
