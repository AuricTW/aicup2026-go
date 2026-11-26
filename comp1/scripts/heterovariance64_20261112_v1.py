"""Candidate support scatter in frozen PLDA space; prospective gates, no refit."""
from pathlib import Path
import sys,argparse,hashlib
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parent))
import metric64_capacity_20261112_v1 as cap
import numpy as np
from threadpoolctl import threadpool_limits
p=cap.p;t=cap.t;q=cap.q;PRIOR=cap.PRIOR
RUN=p.ROOT/'runs/heterovariance64_20261112_v1';SEED=2026111211
PARENT=p.ROOT/'submissions/20261112_v2_metric64_raw768_a050_all_v1/submission_player.csv'


def prepare():
    RUN.mkdir(exist_ok=False);assert p.b.sha(PARENT)=='54eca6b52dd24b044365797c0a663b11b1c2100f6f4b965a9a3ce86b55c94ece';m=p.b.meta();ep=p.b.read(cap.RUN/'EPISODE_screen.json');screenplayers={z['player'] for z in ep['questions']};p.b.write(RUN/'EPISODE_screen.json',ep)
    paths=[PRIOR/'cal_indices.npy',cap.prev.old.RUN/'cal_indices.npy',cap.prev.RUN/'cal_indices.npy',cap.RUN/'cal_indices.npy',p.RUN/'cal_indices.npy',p.DENSE.parent/'validation_indices.npy',p.ROOT/'runs/v2_q1_metric_20261111_v1/fresh_indices.npy',p.ROOT/'runs/v2_q1_fixed_replication1200_20261111_v1/fresh_indices.npy'];old=np.unique(np.concatenate([np.load(f) for f in paths]));blocked={f:set(m[f][old]) for f in p.b.FAMILIES};by=defaultdict(list)
    for i in np.flatnonzero(np.isin(m['owners'].astype(str),ep['candidates'])&(m['hd']<=1)):
        owner=m['owners'][i].decode()
        if owner not in screenplayers and all(m[f][i] not in blocked[f] for f in p.b.FAMILIES):by[owner].append(int(i))
    groups=defaultdict(list)
    for owner,ids in by.items():
        if len(ids)>=5:groups[str(m['ranks'][ids[0]])].append(owner)
    for rank in groups:groups[rank].sort(key=lambda x:hashlib.sha256((str(SEED)+x).encode()).digest())
    order=[];cursor=0
    while any(cursor<len(v) for v in groups.values()):
        for rank in sorted(groups):
            if cursor<len(groups[rank]):order.append(groups[rank][cursor])
        cursor+=1
    banks={};used={f:set() for f in p.b.FAMILIES}
    for owner in order:
        picked=[];local={f:set() for f in p.b.FAMILIES}
        for i in sorted(by[owner],key=lambda i:hashlib.sha256(m['group'][i]+str(SEED).encode()).digest()):
            if any(m[f][i] in used[f] or m[f][i] in local[f] for f in p.b.FAMILIES):continue
            picked.append(i)
            for f in p.b.FAMILIES:local[f].add(m[f][i])
            if len(picked)==5:break
        if len(picked)<5:continue
        banks[owner]=sorted(picked,key=lambda i:m['raw'][i])
        for f in p.b.FAMILIES:used[f].update(local[f])
        if len(banks)==200:break
    assert len(banks)==200,('metadata_capacity',len(banks));conf=dict(ep,questions=[{'id':f'hetero_{n}_{j:04}','indices':banks[owner][:n],'q':n,'target':ep['candidates'].index(owner),'player':owner} for n in (1,2,3,5) for j,owner in enumerate(sorted(banks))],part='confirmation',seed=SEED);p.b.write(RUN/'EPISODE_confirmation.json',conf);fresh=np.asarray(sorted(i for b in banks.values() for i in b),np.int64);cal=np.asarray(sorted(set(np.concatenate(ep['supports']).tolist())|set(fresh.tolist())),np.int64)
    for name,a in [('replay_indices',fresh),('cal_indices',cal)]:np.save(RUN/f'{name}.npy',a,allow_pickle=False)
    for name in ('train_indices.npy','clean_proxy/fit_indices.npy','clean_proxy/H8_fit_indices.npy'):
        fit=np.load(PRIOR/name);assert not set(m['owners'][fit])&set(m['owners'][cal])
        for f in p.b.FAMILIES:assert not set(m[f][fit])&set(m[f][cal])
    protected=p.b.read(cap.RUN/'PROTECTED_INPUTS_SHA256.json');protected.update(p.b.read(cap.RUN/'NEW_MODELS_SHA256.json'));protected[str(PARENT)]=p.b.sha(PARENT)
    for file,h in protected.items():assert p.b.sha(file)==h,file
    p.b.write(RUN/'PROTECTED_INPUTS_SHA256.json',protected)
    pass  # leaderboard record removed for the public release
    p.b.write(RUN/'PROTOCOL.json',{'created_utc':p.b.base.utc(),'source_SHA256':p.b.sha(__file__),'metadata_seed':SEED,'bootstrap_seed':cap.SEED,'hypothesis':'candidate-specific between-game scatter in provided10supportgames maycarry information lostbymean+count, improve uncertainty onfewgames','distinct_from_historical':['oldpergamepredictivemixtures use fixedsharednoise','jointB/WGaussian fits sharedcolor covariance rather than perplayer supportscatter'],'models_frozen':['raw256_v64','raw768_v64'],'formula':'v=clip((nu+support_SSE)/(nu+n-1),0.25,4);postvar=psi*v/(v+n*psi);postmean=postvar*n/v*supportmean;score=-0.5sum(log(v/q+postvar)+(qmean-postmean)^2/(v/q+postvar)+(q-1)*log(v)+query_SSE/v), up toqueryconstant','support_statistics_only':'providedcandidate labels allowedbytask; no query labels, globaltestfit orcrossquestionadaptation','nu':[10,30,100],'neural_term':'equalRRF60 acrosshetero256+hetero768','alphas':[.5,1.],'routes':['Q1','all'],'configs':12,'baseline':'samefrozen cleanprior + finalneuralweight0.4Q1/0.7Q235 times equalRRF256/768; deployedmetric768rollback retained','gates_both_stages':p.b.read(cap.RUN/'PROTOCOL.json')['gates_both_stages'],'selection':'highestpassing4lengthmacroSID,tie fewerroute thenlargernu thenkey; ONEwinner confirmation, no runnerups or newbank afterfailedconfirmation','screen':'same1233 previouslyexposedscreenqueries, countsunequal so macrobylength; 300players; bootstrapbyplayer','confirmation':'200otherplayers,1000newuniquegames,800nestedquestions; sameexposed600 andfixedsupport; NOTgloballyindependentidentityholdout','fresh_family_exclusions':list(map(str,paths)),'no_new_trainfit':True,'engineering_before_quality':['v=1 reduces tooriginalPLDA up toqueryconstants','jointGaussian analytic formula against direct multivariate covariance','coldwarm andquery/support/candidateorder stability'],'if_fail':'no failed candidateexport; keepnegativebranch','original_sealed400_not_opened':True,'Public_answer_key_not_loaded':True,'no_upload':True,'compute':'localCPU4 forstatistics; 4090onlynew1000featuregames afterscreenpassesandnativeidlecheck; respectuserwaitforCompetition2'})
    p.b.write(RUN/'SCOPE_AUDIT.json',{'screen_players':300,'confirmation_players':200,'screen_confirm_target_overlap':0,'new_uniquegames':1000,'all_fit_owner_andfourfamily_overlap':0,'same_exposed600_not_independent':True,'oldmodels_preserved':True,'safe_metadata_SHA256':p.b.sha(p.b.SAFE)});p.b.log(stage='hetero_prepared',models_fitted=0,configs=12)


def stats(z,sets):
    mean,count=p.summarize(z,sets);sse=np.stack([np.sum((z[sorted(s)]-mean[i])**2,axis=0) for i,s in enumerate(sets)]);return mean,count,sse


def predict_gaussian(model,sm,sn,ss,qm,qn,qs,nu):
    v=np.clip((nu+ss)/(nu+sn[:,None]-1),.25,4.) if np.isfinite(nu) else np.ones_like(sm)
    pv=model.psi[None]*v/(v+sn[:,None]*model.psi[None]);pm=pv*sn[:,None]/v*sm;out=np.empty((len(qm),len(sm)),np.float64)
    for i,(mu,n) in enumerate(zip(qm,qn)):
        var=pv+v/n;diff=mu-pm;out[i]=-.5*np.sum(np.log(var)+diff*diff/var+(n-1)*np.log(v)+qs[i]/v,axis=1)
    assert np.isfinite(out).all();return out


def components(raw,ids,ep,nus=(10,30,100),engineering=False):
    order=np.argsort(ids,kind='stable');ids=ids[order];raw=raw[order];pos={int(i):j for j,i in enumerate(ids)};ss=[[pos[i] for i in s] for s in ep['supports']];qs=[[pos[i] for i in v['indices']] for v in ep['questions']];out={nu:[] for nu in nus};errors=[]
    for folder in (PRIOR/'models/raw256_v64',cap.RUN/'models/raw768'):
        x=cap.inputs(raw,'raw');model=p.PLDA(folder/'PLDA.npz');z=model.transform(x);sm,sn,sse=stats(z,ss);qm,qn,qse=stats(z,qs)
        if engineering:
            control=predict_gaussian(model,sm,sn,sse,qm[:8],qn[:8],qse[:8],np.inf);old=model.score(qm[:8],qn[:8],sm,sn);error=float(np.abs(control+.5*qse[:8].sum(1)[:,None]-old).max());assert error<1e-9;errors.append(error)
        for nu in nus:out[nu].append(predict_gaussian(model,sm,sn,sse,qm,qn,qse,nu))
    return {nu:.5/(60+p.b.ranks(a[0]))+.5/(60+p.b.ranks(a[1])) for nu,a in out.items()},errors


def engineering():
    from scipy.stats import multivariate_normal
    from types import SimpleNamespace
    rng=np.random.default_rng(SEED);model=SimpleNamespace(psi=np.asarray([.5,1.,2.]));z=rng.normal(size=(30,3));ss=[list(range(10)),list(range(10,20))];qs=[list(range(20,21)),list(range(21,26))];sm,sn,sse=stats(z,ss);qm,qn,qse=stats(z,qs);nu=30;value=predict_gaussian(model,sm,sn,sse,qm,qn,qse,nu);errors=[]
    for i,qids in enumerate(qs):
        for j in range(2):
            v=np.clip((nu+sse[j])/(nu+sn[j]-1),.25,4);pv=model.psi*v/(v+sn[j]*model.psi);pm=pv*sn[j]/v*sm[j];direct=0.
            for d in range(3):
                cov=v[d]*np.eye(len(qids))+pv[d]*np.ones((len(qids),len(qids)));direct+=multivariate_normal.logpdf(z[qids,d],mean=np.repeat(pm[d],len(qids)),cov=cov)
            direct+=.5*3*np.log(len(qids))+.5*len(qids)*3*np.log(2*np.pi);errors.append(float(abs(value[i,j]-direct)))
    assert max(errors)<1e-10;p.b.write(RUN/'ENGINEERING_FORMULA.json',{'direct_joint_normal_maxabs':max(errors),'passed':True,'before_quality':True,'derivation':'Gaussian posterior predictive with empiricalshrinkwithinvariance, queryconstant discarded','not_integratedStudentT':True})


def blend(base,oldneural,newneural,ep,alpha,route):
    cand=base+cap.countweights(ep)*alpha*(newneural-oldneural);take=np.asarray([route=='all' or z['q']==1 for z in ep['questions']]);out=base.copy();out[take]=cand[take];return out


def screen():
    assert p.b.sha(__file__)==p.b.read(RUN/'PROTOCOL.json')['source_SHA256'];engineering();ep=p.b.read(RUN/'EPISODE_screen.json');parts={nu:[] for nu in (10,30,100)};errors=[]
    for source,root in [('Q1',PRIOR),('multi',cap.prev.old.RUN)]:
        at=[z['source_index'] for z in ep['questions'] if z['source']==source];ee=p.b.read(root/'EPISODE_screen.json');ee=dict(ee,questions=[ee['questions'][i] for i in at]);vv,er=components(np.load(root/'views/cal64.npy',mmap_mode='r'),np.load(root/'cal_indices.npy'),ee,engineering=True);errors+=er
        for nu,v in vv.items():parts[nu].append(v)
    oldg=np.load(cap.RUN/'screen/global64.npy');raw768=np.load(cap.RUN/'screen/raw768.npy');oldneural=.5/(60+p.b.ranks(oldg))+.5/(60+p.b.ranks(raw768));prior=np.load(cap.RUN/'screen/prior.npy');w=cap.countweights(ep);base=(1-w)*prior+w*oldneural;dest=RUN/'screen';dest.mkdir(exist_ok=False);np.save(dest/'baseline.npy',base,allow_pickle=False);np.save(dest/'oldneural.npy',oldneural,allow_pickle=False);results={};passing=[]
    p.b.write(RUN/'ENGINEERING_HOMOSCEDASTIC_CONTROL.json',{'maxabs_up_toqueryconstant':max(errors),'passed':True,'originalPLDA_method_control':True})
    for nu,vs in parts.items():
        v=np.concatenate(vs);np.save(dest/f'nu{nu}.npy',v,allow_pickle=False)
        for alpha in (.5,1.):
            for route in ('Q1','all'):
                score=blend(base,oldneural,v,ep,alpha,route);r=cap.compare(score,base,ep,route);key=f'nu{nu}_a{alpha:g}_{route}';r.update({'nu':nu,'alpha':alpha,'route':route});results[key]=r
                if r['passed']:passing.append(key)
    winner=sorted(passing,key=lambda k:(-results[k]['macro_score_all'],1 if results[k]['route']=='Q1' else 4,-results[k]['nu'],k))[0] if passing else None;p.b.write(RUN/'SCREEN_RESULTS.json',results);p.b.write(RUN/'WINNER_FREEZE.json',{'winner':winner,'selected':results[winner] if winner else None,'passing':passing,'confirmation_quality_unread':True});p.b.log(stage='hetero_screen',winner=winner,passing=passing,summary={k:{'delta':v['delta'],'byq':v['by_games_delta'],'Top1net':v['Top1_net'],'CI95':v['CI95']} for k,v in results.items()})


def assemble():
    root=RUN/'dev_results';receipt=p.b.read(root/'RECEIPT.json');assert receipt['all_valid'] and not receipt['fit_or_optimizer_used']
    for f,h in receipt['SHA256'].items():assert p.b.sha(root/f)==h
    ids=np.load(RUN/'cal_indices.npy');oldids=np.load(PRIOR/'cal_indices.npy');newids=np.load(root/'indices.npy');assert np.array_equal(newids,np.load(RUN/'replay_indices.npy'));pos={int(i):('old',j) for j,i in enumerate(oldids)};pos.update({int(i):('new',j) for j,i in enumerate(newids)});dest=RUN/'views';dest.mkdir(exist_ok=False)
    for v in (8,64):
        bank={'old':np.load(PRIOR/('views/cal_original8.npy' if v==8 else 'views/cal64.npy'),mmap_mode='r'),'new':np.load(root/f'view{v}.npy',mmap_mode='r')};np.save(dest/f'cal{v}.npy',np.stack([bank[pos[int(i)][0]][pos[int(i)][1]] for i in ids]),allow_pickle=False)


def confirm():
    sel=p.b.read(RUN/'WINNER_FREEZE.json')['selected'];assert sel;ep=p.b.read(RUN/'EPISODE_confirmation.json');ids=np.load(RUN/'cal_indices.npy');m=p.b.meta();pos={int(i):j for j,i in enumerate(ids)};ss=[[pos[i] for i in s] for s in ep['supports']];qs=[[pos[i] for i in z['indices']] for z in ep['questions']];raw=np.asarray(np.load(p.ROOT/'data/processed/features.npy',mmap_mode='r')[m['rows'][ids]],np.float32);hist=t.cp.histograms(raw,p.b.read(p.ROOT/'data/processed/feature_schema.json'),sqrt=True);codes=p.b.sequence_export(ids,m);gm=p.PLDA(PRIOR/'clean_proxy/models/global_PLDA.npz');z=gm.transform(hist);sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);gs=gm.score(qm,qn,sm,sn);cs=t.cp.color_score(raw,m['colors'][ids],ss,qs,PRIOR/'clean_proxy/models');seq=t.clean.sequence_scores(codes,m['colors'][ids],ss,qs);fc=t.cp.fixed_rrf60(t.cp.fixed_rrf60(gs,cs),seq);hm=p.PLDA(PRIOR/'clean_proxy/models/H8/PLDA.npz');z=hm.transform(q.oldtrain.pre(np.load(RUN/'views/cal8.npy'),PRIOR/'clean_proxy/models/H8'));sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);hs=hm.score(qm,qn,sm,sn);prior=.75*fc+.25/(60+p.b.ranks(hs));raw64=np.load(RUN/'views/cal64.npy',mmap_mode='r');g=t.component(PRIOR/'models/raw256_v64',raw64,ids,[ep])[0];g768=cap.component(cap.RUN/'models/raw768',raw64,ids,ep);oldneural=.5/(60+p.b.ranks(g))+.5/(60+p.b.ranks(g768));w=cap.countweights(ep);base=(1-w)*prior+w*oldneural;vv,_=components(raw64,ids,ep,nus=(sel['nu'],));newneural=vv[sel['nu']];score=blend(base,oldneural,newneural,ep,sel['alpha'],sel['route']);r=cap.compare(score,base,ep,sel['route']);r.update({'nu':sel['nu'],'alpha':sel['alpha'],'route':sel['route'],'winner':p.b.read(RUN/'WINNER_FREEZE.json')['winner'],'same_exposed600_not_independent':True,'new1000games_fourfamilies_disjoint_namedbanks':True,'screen300_confirmation200_targetdisjoint':True,'clean_proxy_not_exactproduction':True});dest=RUN/'confirmation';dest.mkdir(exist_ok=False)
    for name,a in [('baseline',base),('candidate',score),('oldneural',oldneural),('newneural',newneural)]:np.save(dest/f'{name}.npy',a,allow_pickle=False)
    p.b.write(RUN/'CONFIRMATION_RESULT.json',r);p.b.log(stage='hetero_confirmation',**r)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','screen','assemble','confirm']);v=parser.parse_args()
    with threadpool_limits(limits=4):{'prepare':prepare,'screen':screen,'assemble':assemble,'confirm':confirm}[v.stage]()
