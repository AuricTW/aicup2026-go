"""Train-only capacity/radial-normalization pilot; protect official weight070."""
from pathlib import Path
import sys,argparse,hashlib
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parent))
import postQ235_aggregation_20261111_v1 as prev
import numpy as np
from threadpoolctl import threadpool_limits
p=prev.p;t=prev.t;q=prev.q;PRIOR=prev.PRIOR
RUN=p.ROOT/'runs/metric64_capacity_20261112_v1';SEED=2026111201
PARENT=p.ROOT/'submissions/20261111_v2_Q235_weight070_v1/submission_player.csv'
MODELS=[('raw',384),('raw',512),('raw',768),('radial',256),('radial',512)]


def prepare():
    RUN.mkdir(exist_ok=False);assert p.b.sha(PARENT)=='2a865c830b4ead4f814af32fa5beb1363cc5b7dce15695a1198211d0a65d3172';m=p.b.meta();a=p.b.read(PRIOR/'EPISODE_screen.json');b=p.b.read(prev.old.RUN/'EPISODE_screen.json');chosen=sorted({z['player'] for z in a['questions']},key=lambda x:hashlib.sha256((str(SEED)+x).encode()).digest())[:300];chosen=set(chosen)
    questions=[dict(z,source='Q1',source_index=i) for i,z in enumerate(a['questions']) if z['player'] in chosen]+[dict(z,source='multi',source_index=i) for i,z in enumerate(b['questions']) if z['player'] in chosen];ep=dict(a,questions=questions,part='screen',seed=SEED);p.b.write(RUN/'EPISODE_screen.json',ep);assert {z['player'] for z in questions}==chosen and sum(z['q']==1 for z in questions)==900
    paths=[PRIOR/'cal_indices.npy',prev.old.RUN/'cal_indices.npy',prev.RUN/'cal_indices.npy',p.RUN/'cal_indices.npy',p.DENSE.parent/'validation_indices.npy',p.ROOT/'runs/v2_q1_metric_20261111_v1/fresh_indices.npy',p.ROOT/'runs/v2_q1_fixed_replication1200_20261111_v1/fresh_indices.npy'];old=np.unique(np.concatenate([np.load(f) for f in paths]));blocked={f:set(m[f][old]) for f in p.b.FAMILIES};by=defaultdict(list)
    for i in np.flatnonzero(np.isin(m['owners'].astype(str),a['candidates'])&(m['hd']<=1)):
        owner=m['owners'][i].decode()
        if owner not in chosen and all(m[f][i] not in blocked[f] for f in p.b.FAMILIES):by[owner].append(int(i))
    groups=defaultdict(list)
    for owner,ids in by.items():
        if len(ids)>=5:groups[str(m['ranks'][ids[0]])].append(owner)
    for rank in groups:groups[rank].sort(key=lambda x:hashlib.sha256(('confirm|'+str(SEED)+x).encode()).digest())
    ordered=[];cursor=0
    while any(cursor<len(v) for v in groups.values()):
        for rank in sorted(groups):
            if cursor<len(groups[rank]):ordered.append(groups[rank][cursor])
        cursor+=1
    banks={};used={f:set() for f in p.b.FAMILIES}
    for owner in ordered:
        selected=[];local={f:set() for f in p.b.FAMILIES}
        for i in sorted(by[owner],key=lambda i:hashlib.sha256(m['group'][i]+str(SEED).encode()).digest()):
            if any(m[f][i] in used[f] or m[f][i] in local[f] for f in p.b.FAMILIES):continue
            selected.append(i)
            for f in p.b.FAMILIES:local[f].add(m[f][i])
            if len(selected)==5:break
        if len(selected)<5:continue
        banks[owner]=sorted(selected,key=lambda i:m['raw'][i])
        for f in p.b.FAMILIES:used[f].update(local[f])
        if len(banks)==200:break
    assert len(banks)==200,('confirmation_capacity',len(banks));conf=dict(a,questions=[{'id':f'm64_{n}_{j:04}','indices':banks[o][:n],'q':n,'target':a['candidates'].index(o),'player':o} for n in (1,2,3,5) for j,o in enumerate(sorted(banks))],part='confirmation',seed=SEED);p.b.write(RUN/'EPISODE_confirmation.json',conf)
    fresh=np.asarray(sorted(i for v in banks.values() for i in v),np.int64);cal=np.asarray(sorted(set(np.concatenate(a['supports']).tolist())|set(fresh.tolist())),np.int64);assert len(cal)==7000 and len(fresh)==1000
    for name,arr in [('replay_indices',fresh),('cal_indices',cal)]:np.save(RUN/f'{name}.npy',arr,allow_pickle=False)
    excluded=np.flatnonzero(np.isin(m['owners'].astype(str),a['candidates']))
    for name in ('train_indices.npy','clean_proxy/fit_indices.npy','clean_proxy/H8_fit_indices.npy'):
        fit=np.load(PRIOR/name);assert not set(m['owners'][fit])&set(m['owners'][excluded])
        for f in p.b.FAMILIES:assert not set(m[f][fit])&set(m[f][excluded]),(name,f)
    train=np.load(PRIOR/'train_indices.npy');labels=np.load(PRIOR/'train_labels.npy');assert len(train)==16384 and len(set(m['owners'][train]))==512 and np.array_equal(labels,np.repeat(np.arange(512),32))
    assert all(np.sum(m['colors'][train[j*32:(j+1)*32]]==0)==16 for j in range(512))
    protected=p.b.read(prev.RUN/'PROTECTED_INPUTS_SHA256.json');protected[str(PARENT)]=p.b.sha(PARENT)
    pinned=p.b.read(PRIOR/'FINAL_MANIFEST_SHA256.json')
    for f in (PRIOR/'views/train64.npy',PRIOR/'train_indices.npy',PRIOR/'train_labels.npy'):protected[str(f)]=pinned[str(f.relative_to(p.ROOT))]
    p.b.write(RUN/'PROTECTED_INPUTS_SHA256.json',protected)
    for file,h in protected.items():assert p.b.sha(file)==h
    feedback={};p.b.write(RUN/'OFFICIAL_FEEDBACK.json',feedback)
    p.b.write(RUN/'PROTOCOL.json',{'created_utc':p.b.base.utc(),'source_SHA256':p.b.sha(__file__),'seed':SEED,'hypotheses':['64point summary may retain additional identity directions beyond256 PCA dimensions','pergame radial magnitude normalization may improve Gaussian metric robustness'],
      'models':[{'kind':k,'dimension':d} for k,d in MODELS],'train':'same16384games,512players,16B+16W; sameHumanSL64 featurebank; allfull600owners_and4families excluded','radial':'existingtrain-onlystandardization; rowL2 normalization times sqrt2339 BEFORE fittingPLDA; no testfit; analogous inspiration notKaldi reproduction','PLDA':'sameexistingfitter, shrinkage0.1; newfolder only; no external weights/code/data','modelcontrol':'raw256_v64 frozen; raw512 vsradial512 isolates normalizer; raw256 vsradial256 also available',
      'baseline':'cleanproxy prior, finalneuralweights0.4Q1/0.7Q235; exact deployedweight070 remains rollback','fusion':'retain percountweight andprior, replacealpha0.5or1 ofneuralRRF60 withnewmodelRRF60','alphas':[.5,1.],'routes':['Q1','all'],'configs':20,'screen':'300oldQ1screen targets,900Q1; onlymultiscreen querieswhoseplayers areinthese300; countsunequal soequalweightmeanof4game-lengthmeans','fresh_confirmation':'200otherplayers and1000newuniquegames,800nested1/2/3/5questions;sameexposed600identities andsupport10;notpristineidentityvalidation',
      'exclusion_sources':list(map(str,paths)),'gates_both_stages':{'routed_macro_SID_delta_min':{'Q1':.015,'all':.0075},'Top1_net_min':'max(4 ifQ1else6,ceil(0.01*routed_n))','each_routed_length_delta_min':0,'5000_sharedplayer_cluster95CI_lower_gt':0},'bootstrap_seed':SEED,'bootstrap':'resampleplayers and recompute perlengthmeans, thenmacro; no800independent assumption','selection':'highestpassingmacroSIDover4lengths, tie fewer routedlengths then lowerdimension thenkey; confirmONEwinner only; no runnerups/newbanks afterfailedconfirmation','if_none':'retainnegativebranch; do not openconfirmationquality or export it','original_sealed400_not_opened':True,'Public_answer_key_not_loaded':True,'no_upload':True,'fit_compute':'localCPU4threads, cachedfrozenfeatureonly','freshfeature_compute':'4090ifidle, sameworker fixed32FP16, cap5400s, only1000newgames afterscreenpasses'})
    p.b.write(RUN/'RESEARCH_SOURCES.json',{'checked_Taipei_date':'2026-11-12','inspiration':[{'title':'Garcia-Romero and Espy-Wilson, INTERSPEECH2011: Analysis of i-vector length normalization','url':'https://www.isca-archive.org/interspeech_2011/garciaromero11_interspeech.html','scope':'speakerrecognition, motivates a local hypothesis; its reported gains are NOT Go gains'},{'title':'Kaldi Plda class reference','url':'https://kaldi-asr.org/doc/classkaldi_1_1Plda.html','scope':'normalization and PLDA reference only, no code copied, not implementation-equivalent'}],'new_external_assets':False,'new_method_code':'local derivation and existing localPLDA reused'})
    p.b.write(RUN/'SCOPE_AUDIT.json',{'screen_players':300,'screen_questions':len(questions),'screen_by_games':{str(n):sum(z['q']==n for z in questions) for n in (1,2,3,5)},'confirmation_players':200,'confirmation_questions':800,'new_unique_query_games':1000,'screen_confirm_target_overlap':0,'all_full600fit_owner_and4family_overlap':0,'original_sealed400_not_opened':True,'safe_SHA256':p.b.sha(p.b.SAFE),'same_exposed600_not_independent':True});p.b.log(stage='metric64_prepared',screen_n=len(questions),confirm_n=800,models=len(MODELS))


def inputs(raw,kind):
    x=q.oldtrain.pre(raw,PRIOR/'models/raw256_v64')
    if kind=='radial':
        xx=np.asarray(x,np.float64);norm=np.sqrt(np.sum(xx*xx,axis=1));x=(xx*(np.sqrt(x.shape[1])/np.maximum(norm,1e-12))[:,None]).astype(np.float32)
    assert np.isfinite(x).all();return x


def train():
    assert p.b.sha(__file__)==p.b.read(RUN/'PROTOCOL.json')['source_SHA256'];raw=np.load(PRIOR/'views/train64.npy',mmap_mode='r');labels=np.load(PRIOR/'train_labels.npy');dest=RUN/'models';dest.mkdir(exist_ok=False)
    for kind,dim in MODELS:
        folder=dest/f'{kind}{dim}';folder.mkdir();r=p.fit_plda(inputs(raw,kind),labels,folder/'PLDA.npz',dimension=dim);r.update({'kind':kind,'no_cal_or_test_labels_in_fit':True,'sameaudited512train':True});p.b.write(folder/'FIT_RECEIPT.json',r);p.b.log(stage='metric64_fit',model=folder.name,**r)
    p.b.write(dest/'COMPLETION.json',{'all_fits_complete':True,'no_quality_read':True})


def component(folder,raw,ids,ep):
    order=np.argsort(ids,kind='stable');ids=ids[order];raw=raw[order];model=p.PLDA(folder/'PLDA.npz');z=model.transform(inputs(raw,folder.name.rstrip('0123456789')));pos={int(i):j for j,i in enumerate(ids)};ss=[[pos[i] for i in s] for s in ep['supports']];qs=[[pos[i] for i in v['indices']] for v in ep['questions']];sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);value=model.score(qm,qn,sm,sn);take=np.arange(0,len(qs),53)[::-1];assert np.array_equal(value[take],model.score(qm[take],qn[take],sm,sn));return value


def countweights(ep):return np.asarray([.4 if z['q']==1 else .7 for z in ep['questions']])[:,None]
def baseline(prior,global64,ep):
    w=countweights(ep);return (1-w)*prior+w/(60+p.b.ranks(global64))
def fuse(base,g,new,ep,alpha,route):
    w=countweights(ep);candidate=base+w*alpha*(1/(60+p.b.ranks(new))-1/(60+p.b.ranks(g)));mask=np.asarray([route=='all' or z['q']==1 for z in ep['questions']]);score=base.copy();score[mask]=candidate[mask];return score


def compare(score,base,ep,route):
    av,ar=t.values(score,ep);bv,br=t.values(base,ep);delta=av-bv;counts=np.asarray([z['q'] for z in ep['questions']]);lengths=(1,) if route=='Q1' else (1,2,3,5);mask=np.isin(counts,lengths);players=sorted({z['player'] for z in ep['questions'] if z['q'] in lengths});pos={v:i for i,v in enumerate(players)};sums=np.zeros((len(players),len(lengths)));nn=np.zeros_like(sums)
    for z,v in zip(ep['questions'],delta):
        if z['q'] in lengths:i=pos[z['player']];j=lengths.index(z['q']);sums[i,j]+=v;nn[i,j]+=1
    draws=np.random.default_rng(SEED).integers(len(players),size=(5000,len(players)));den=nn[draws].sum(1);assert np.all(den>0);boot=(sums[draws].sum(1)/den).mean(1);ci=np.quantile(boot,[.025,.975]).tolist();by={str(n):float(delta[counts==n].mean()) for n in (1,2,3,5)};mean=float(np.mean([by[str(n)] for n in lengths]));net=int((ar[mask]==1).sum()-(br[mask]==1).sum());minimum=max(4 if route=='Q1' else 6,int(np.ceil(.01*mask.sum())));checks={'macro_mean':mean>=(.015 if route=='Q1' else .0075),'Top1':net>=minimum,'CI':ci[0]>0,'each_routed_length':all(by[str(n)]>=0 for n in lengths)}
    return {'delta':mean,'CI95':ci,'by_games_delta':by,'Top1_net':net,'Top1_required':minimum,'Top1':int((ar[mask]==1).sum()),'baseline_Top1':int((br[mask]==1).sum()),'Top5':int((ar[mask]<=5).sum()),'baseline_Top5':int((br[mask]<=5).sum()),'n_routed':int(mask.sum()),'players_routed':len(players),'score_routed_macro':float(np.mean([av[counts==n].mean() for n in lengths])),'baseline_routed_macro':float(np.mean([bv[counts==n].mean() for n in lengths])),'macro_score_all':float(np.mean([av[counts==n].mean() for n in (1,2,3,5)])),'baseline_macro_all':float(np.mean([bv[counts==n].mean() for n in (1,2,3,5)])),'checks':checks,'passed':all(checks.values())}


def screen():
    assert p.b.sha(__file__)==p.b.read(RUN/'PROTOCOL.json')['source_SHA256'];ep=p.b.read(RUN/'EPISODE_screen.json');priors=[];globalscores=[];components={f'{k}{d}':[] for k,d in MODELS}
    for source,root in [('Q1',PRIOR),('multi',prev.old.RUN)]:
        at=[z['source_index'] for z in ep['questions'] if z['source']==source];ee=p.b.read(root/'EPISODE_screen.json');ee=dict(ee,questions=[ee['questions'][i] for i in at]);ids=np.load(root/'cal_indices.npy');raw=np.load(root/'views/cal64.npy',mmap_mode='r');g=t.component(PRIOR/'models/raw256_v64',raw,ids,[ee])[0];globalscores.append(g)
        if source=='Q1':priors.append(np.load(PRIOR/'parent/prior_screen.npy')[at])
        else:priors.append((np.load(root/'predictions/screen/replacement.npy')[at]-.4/(60+p.b.ranks(g)))/.6)
        for name in components:components[name].append(component(RUN/'models'/name,raw,ids,ee))
    prior=np.concatenate(priors);g=np.concatenate(globalscores);base=baseline(prior,g,ep);dest=RUN/'screen';dest.mkdir(exist_ok=False)
    for name,a in [('prior',prior),('global64',g),('baseline',base)]:np.save(dest/f'{name}.npy',a,allow_pickle=False)
    results={};passing=[]
    for name,parts in components.items():
        v=np.concatenate(parts);np.save(dest/f'{name}.npy',v,allow_pickle=False)
        for alpha in (.5,1.):
            for route in ('Q1','all'):
                score=fuse(base,g,v,ep,alpha,route);r=compare(score,base,ep,route);key=f'{name}_a{alpha:g}_{route}';r.update({'model':name,'alpha':alpha,'route':route});results[key]=r
                if r['passed']:passing.append(key)
    winner=sorted(passing,key=lambda k:(-results[k]['macro_score_all'],1 if results[k]['route']=='Q1' else 4,int(results[k]['model'].lstrip('radial').lstrip('w')),k))[0] if passing else None
    p.b.write(RUN/'SCREEN_RESULTS.json',results);p.b.write(RUN/'WINNER_FREEZE.json',{'winner':winner,'selected':results[winner] if winner else None,'passing':passing,'new_confirmation_quality_unread':True});p.b.log(stage='metric64_screen',winner=winner,passing=passing,summary={k:{'delta':v['delta'],'CI95':v['CI95'],'byq':v['by_games_delta'],'Top1net':v['Top1_net']} for k,v in results.items()})


def assemble():
    root=RUN/'dev_results';receipt=p.b.read(root/'RECEIPT.json');assert receipt['all_valid'] and not receipt['fit_or_optimizer_used']
    for f,h in receipt['SHA256'].items():assert p.b.sha(root/f)==h
    newids=np.load(root/'indices.npy');assert np.array_equal(newids,np.load(RUN/'replay_indices.npy'));oldids=np.load(PRIOR/'cal_indices.npy');ids=np.load(RUN/'cal_indices.npy');pos={int(i):('old',j) for j,i in enumerate(oldids)};pos.update({int(i):('new',j) for j,i in enumerate(newids)});dest=RUN/'views';dest.mkdir(exist_ok=False)
    for v in (8,32,64):
        data={'old':np.load(PRIOR/('views/cal_original8.npy' if v==8 else f'views/cal{v}.npy'),mmap_mode='r'),'new':np.load(root/f'view{v}.npy',mmap_mode='r')};np.save(dest/f'cal{v}.npy',np.stack([data[pos[int(i)][0]][pos[int(i)][1]] for i in ids]),allow_pickle=False)


def confirm():
    freeze=p.b.read(RUN/'WINNER_FREEZE.json');assert freeze['winner'];sel=freeze['selected'];ep=p.b.read(RUN/'EPISODE_confirmation.json');ids=np.load(RUN/'cal_indices.npy');m=p.b.meta();pos={int(i):j for j,i in enumerate(ids)};ss=[[pos[i] for i in s] for s in ep['supports']];qs=[[pos[i] for i in z['indices']] for z in ep['questions']];raw=np.asarray(np.load(p.ROOT/'data/processed/features.npy',mmap_mode='r')[m['rows'][ids]],np.float32);hist=t.cp.histograms(raw,p.b.read(p.ROOT/'data/processed/feature_schema.json'),sqrt=True);codes=p.b.sequence_export(ids,m)
    gm=p.PLDA(PRIOR/'clean_proxy/models/global_PLDA.npz');z=gm.transform(hist);sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);gs=gm.score(qm,qn,sm,sn);cs=t.cp.color_score(raw,m['colors'][ids],ss,qs,PRIOR/'clean_proxy/models');seq=t.clean.sequence_scores(codes,m['colors'][ids],ss,qs);fc=t.cp.fixed_rrf60(t.cp.fixed_rrf60(gs,cs),seq);hm=p.PLDA(PRIOR/'clean_proxy/models/H8/PLDA.npz');z=hm.transform(q.oldtrain.pre(np.load(RUN/'views/cal8.npy'),PRIOR/'clean_proxy/models/H8'));sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);h=hm.score(qm,qn,sm,sn);prior=.75*fc+.25/(60+p.b.ranks(h));raw64=np.load(RUN/'views/cal64.npy',mmap_mode='r');g=t.component(PRIOR/'models/raw256_v64',raw64,ids,[ep])[0];v=component(RUN/'models'/sel['model'],raw64,ids,ep);base=baseline(prior,g,ep);score=fuse(base,g,v,ep,sel['alpha'],sel['route']);r=compare(score,base,ep,sel['route']);r.update({'winner':freeze['winner'],'model':sel['model'],'alpha':sel['alpha'],'route':sel['route'],'same_exposed600_not_independent':True,'new1000games_namedbanks_disjoint':True,'screen300_confirmation200_targetdisjoint':True,'clean_proxy_not_exactproduction':True});dest=RUN/'confirmation';dest.mkdir(exist_ok=False)
    for name,a in [('prior',prior),('global64',g),('component',v),('baseline',base),('candidate',score)]:np.save(dest/f'{name}.npy',a,allow_pickle=False)
    p.b.write(RUN/'CONFIRMATION_RESULT.json',r);p.b.log(stage='metric64_confirmation',**r)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','screen','assemble','confirm']);v=parser.parse_args()
    with threadpool_limits(limits=4):{'prepare':prepare,'train':train,'screen':screen,'assemble':assemble,'confirm':confirm}[v.stage]()
