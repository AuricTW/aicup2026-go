"""Equal-game-budget player-diversity vs repeats on the frozen90 representation."""
from pathlib import Path
import sys,argparse,hashlib
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parent))
import complete90_pilot_20261112_v1 as old
import clean_proxy_fresh600_dense64_20261111_v2 as cleanbasis
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
p=old.p;cap=old.cap;t=old.t;q=old.q
RUN=p.ROOT/'runs/complete90_scale_20261112_v1';SEED=2026111231
PARENT=p.ROOT/'submissions/20261112_v2_complete90_mean_a050_all_v1/submission_player.csv'
BANKS=('repeat512','diverse1024')


def h(value):return hashlib.sha256((str(SEED)+'|'+str(value)).encode()).digest()


def interleave(groups):
    groups={k:sorted(v,key=h) for k,v in groups.items()};out=[];j=0
    while any(j<len(v) for v in groups.values()):
        for k in sorted(groups):
            if j<len(groups[k]):out.append(groups[k][j])
        j+=1
    return out


def unique_pick(indices,m,used,n):
    selected=[];local={f:set() for f in p.b.FAMILIES}
    for i in sorted(map(int,indices),key=lambda i:h(m['group'][i])):
        if any(m[f][i] in used[f] or m[f][i] in local[f] for f in p.b.FAMILIES):continue
        selected.append(i)
        for f in p.b.FAMILIES:local[f].add(m[f][i])
        if len(selected)==n:break
    return selected,local


def prepare():
    RUN.mkdir(exist_ok=False);assert p.b.sha(PARENT)=='f2780f4f700a5def6a285ac366eaf843409a5b9cd29a209cbb5da3564a47507d'
    m=p.b.meta();owners=m['owners'].astype(str);oldtrain=np.load(old.PRIOR/'train_indices.npy');old16=np.load(old.RUN/'train_indices.npy')
    previous=set(p.b.read(old.RUN/'EPISODE_screen.json')['candidates']);earlier=set(p.b.read(p.b.RUN/'calibration/ep0.json')['candidates']);hist=set(owners[np.load(p.DENSE.parent/'validation_indices.npy')]);old875=set(owners[np.load(p.ROOT/'runs/v2_q1_diversity875_20261111_v1/train_indices.npy')]);banned=previous|earlier|hist|old875|set(owners[oldtrain])
    # Whole-owner exclusion for any family occurring in the immutable old metric fit.
    for f in p.b.FAMILIES:banned.update(owners[np.isin(m[f],np.unique(m[f][oldtrain]))])
    by=defaultdict(list)
    for i in np.flatnonzero(~np.isin(owners,sorted(banned))&(m['hd']<=1)):by[owners[i]].append(int(i))
    groups=defaultdict(list)
    for owner,ii in by.items():
        if len(ii)>=15:groups[str(m['ranks'][ii[0]])].append(owner)
    ordered=interleave(groups);used={f:set() for f in p.b.FAMILIES};players=[];banks={}
    for owner in ordered:
        ix,local=unique_pick(by[owner],m,used,15)
        if len(ix)<15:continue
        players.append(owner);banks[owner]=ix
        for f in p.b.FAMILIES:used[f].update(local[f])
        if len(players)==600:break
    assert len(players)==600,('new600_capacity',len(players));players.sort();supports=[banks[o][:10] for o in players]
    targetgroups=defaultdict(list)
    for o in players:targetgroups[str(m['ranks'][banks[o][0]])].append(o)
    targetorder=interleave({k:[o+'|target' for o in v] for k,v in targetgroups.items()});targetorder=[o.rsplit('|',1)[0] for o in targetorder]
    eps=[]
    for part,targets in [('screen',targetorder[:300]),('confirmation',targetorder[300:500])]:
        questions=[{'id':f'scale90_{part}_{n}_{j:04}','indices':banks[o][10:10+n],'q':n,'target':players.index(o),'player':o} for n in (1,2,3,5) for j,o in enumerate(targets)]
        ep={'candidates':players,'supports':supports,'questions':questions,'seed':SEED,'part':part};eps.append(ep);p.b.write(RUN/f'EPISODE_{part}.json',ep)
    cal=np.asarray(sorted({i for s in supports for i in s}|{i for ep in eps for z in ep['questions'] for i in z['indices']}),np.int64);assert len(cal)==8500
    excluded=np.isin(owners,sorted(set(players)|previous|earlier|hist));safe=(~excluded)&(m['hd']<=1)
    for f in p.b.FAMILIES:safe &= ~np.isin(m[f],np.unique(m[f][excluded]))
    assert np.all(safe[oldtrain])
    trainby=defaultdict(lambda:[[],[]])
    for i in np.flatnonzero(safe&~np.isin(owners,sorted(set(owners[oldtrain])))):trainby[owners[i]][int(m['colors'][i])].append(int(i))
    tg=defaultdict(list)
    for o,v in trainby.items():
        if min(map(len,v))>=8:tg[str(m['ranks'][v[0][0]])].append(o)
    extra=[];newplayers=[];tu={f:set(m[f][old16]) for f in p.b.FAMILIES}
    for o in interleave(tg):
        picked=[];ls={f:set() for f in p.b.FAMILIES}
        for c in (0,1):
            pp,local=unique_pick(trainby[o][c],m,{f:tu[f]|ls[f] for f in p.b.FAMILIES},8);picked+=pp
            for f in p.b.FAMILIES:ls[f].update(local[f])
        if len(picked)!=16:continue
        newplayers.append(o);extra+=picked
        for f in p.b.FAMILIES:tu[f].update(ls[f])
        if len(newplayers)==512:break
    assert len(newplayers)==512;extra=np.asarray(extra,np.int64);diverse=np.concatenate((old16,extra));assert len(diverse)==len(oldtrain)==16384
    for name,a in [('train_repeat512',oldtrain),('labels_repeat512',np.repeat(np.arange(512),32)),('train_diverse1024',diverse),('labels_diverse1024',np.repeat(np.arange(1024),16)),('cal_indices',cal)]:np.save(RUN/f'{name}.npy',a,allow_pickle=False)
    for fit in (oldtrain,diverse):
        assert not set(owners[fit])&set(players)
        for f in p.b.FAMILIES:assert not set(m[f][fit])&set(m[f][excluded])
    proxy=RUN/'clean_proxy';proxy.mkdir();oldfit=np.load(old.PRIOR/'clean_proxy/fit_indices.npy');fit=oldfit[safe[oldfit]];h8=np.load(old.PRIOR/'clean_proxy/H8_fit_indices.npy');h8=h8[safe[h8]];assert len(fit)>30000 and len(h8)>256 and len(set(owners[h8]))>128
    np.save(proxy/'fit_indices.npy',fit,allow_pickle=False);np.save(proxy/'H8_fit_indices.npy',h8,allow_pickle=False)
    for ii in (fit,h8):
        assert not set(owners[ii])&set(owners[excluded])
        for f in p.b.FAMILIES:assert not set(m[f][ii])&set(m[f][excluded])
    trainunion=np.union1d(oldtrain,diverse);cached=np.load(old.RUN/'dev_results/indices.npy');needed=np.union1d(trainunion,cal);replay=np.setdiff1d(needed,cached);assert len(replay)==24884
    np.save(RUN/'replay_indices.npy',replay,allow_pickle=False);np.save(RUN/'all_indices.npy',needed,allow_pickle=False)
    protected=p.b.read(old.RUN/'PROTECTED_INPUTS_SHA256.json');protected.update(p.b.read(old.RUN/'MODELS_SHA256.json'));protected[str(PARENT)]=p.b.sha(PARENT)
    for root in (old.RUN/'dev_results',old.RUN/'official_results'):
        for f in root.glob('*.npy'):protected[str(f)]=p.b.sha(f)
    for f in (old.__file__,cap.__file__,cleanbasis.__file__,p.b.SAFE):protected[str(f)]=p.b.sha(f)
    p.b.write(RUN/'PROTECTED_INPUTS_SHA256.json',protected)
    pass  # leaderboard record removed for the public release
    p.b.write(RUN/'PROTOCOL.json',{'created_utc':p.b.base.utc(),'source_SHA256':p.b.sha(__file__),'metadata_seed':SEED,'bootstrap_seed':cap.SEED,'hypothesis':'control frozen90representation andequal16384gamebudget, test repeats512x32 vsdiversity1024x16; paired64controls for each bank','train_banks':{'repeat512':'512x32,16B16W sameoldfulltrain','diverse1024':'old512x16 plusnew512x16,8B8W each,16384games total'},'features':'exact priorcomplete90worker, frozenHumanSL1d,HD0/1,32FP16chunks,2339summary,paired64and90','models':'eachbankx64/90x256/768=8fits,train-onlyscaler,withinshrink0.1; only90equalRRF256/768 candidates','evaluation':'NEW600relativeold875neuraltrain,historical723,earlier600Nov11,recent600; support10; screen300players1200questions,confirmation200differentplayers800questions; nested1/2/3/5,100distractoronly','historical_exposure':'broadlegacyfitexposurepossible; oldopaqueledgernamespace doesnotmatchsafeowners, pristinehistory NOT proven; newallfitsexclude full600identity+4families; cleanproxyrefit required','cleanproxy':'existingcleanstatfitrows filteredfullnew600/families; refit global/B/WPLDA,train-onlyopeningbackground,H8preprocess/PLDA; nolegacyMLP/U0/Gaussian','baseline':'samecomplete90production formula usingnewcleanprior; neural=.5oldfull64ensemble+.5oldhalf90ensemble, weights0.4Q1/0.7Q235','fusion':'onlyreplacealpha of existing90half: base+w*0.5*alpha*(new90ensemble-oldhalf90ensemble); keepother64half,prior,countweights','banks':list(BANKS),'alphas':[.5,1.],'routes':['Q1','all'],'configs':8,'gates_both_stages':p.b.read(cap.RUN/'PROTOCOL.json')['gates_both_stages'],'additional_gate':'new90ensemble routedSID>=samebank64ensemble; pairedCI reported, not requiredpositive','selection':'highestpassing4lengthmacroSID,tie fewerroute,smaller alpha,banklexical; ONEwinner confirmation, no runnerup/newbank afterfailedconfirmation','confirmation_quality_unread':True,'if_fail':'retainallnegativeevidence,no failed candidateexport','source_original_sealed400_excluded_upstream':True,'original_sealed400_notopened':True,'Public_answer_key_notloaded':True,'no_upload':True,'compute':'CPU4fit,4090idleonly,candidatefeaturejob<=5400s,8CPUworkers; nonewexternalassets orenvchange','reuse':'8192old90train fromhashboundcache; replay24884newgames, officialfeaturesreuseexactworker14200cache aftergates','engineering':'sameold8/64nativeparity,coldwarmreverse probes; sourceSHA identical, no postquality feature rulechange'})
    scope={'safe_rows':len(owners),'safe_players':len(set(owners)),'new600_recentoverlap':len(set(players)&previous),'new600_old875overlap':len(set(players)&old875),'new600_hist723overlap':len(set(players)&hist),'new600_earlier600overlap':len(set(players)&earlier),'all_fit_full600_owner_and4family_overlap':0,'trainbank_games':16384,'new_neuraltrain512':len(newplayers),'train_union_games':len(trainunion),'cached_reused':len(needed)-len(replay),'replay_games':len(replay),'validation_games':len(cal),'proxy_histfit':len(fit),'proxy_H8fit':len(h8),'screen300_confirm200_targetdisjoint':True,'historical_pristine_not_proven':True,'original_sealed400_notopened':True,'safe_metadata_SHA256':p.b.sha(p.b.SAFE)};p.b.write(RUN/'SCOPE_AUDIT.json',scope);p.b.log(stage='complete90_scale_prepared',**scope)


def build_proxy():
    m=p.b.meta();dest=RUN/'clean_proxy';ids=np.load(dest/'fit_indices.npy');oldids=np.load(old.PRIOR/'clean_proxy/fit_indices.npy');oldx=np.load(old.PRIOR/'clean_proxy/hist_fit.npy',mmap_mode='r');at=np.searchsorted(oldids,ids);assert np.array_equal(oldids[at],ids);folder=dest/'models';folder.mkdir(exist_ok=False);x=oldx[at]
    r=p.fit_plda(x,m['owners'][ids],folder/'global_PLDA.npz',dimension=256);p.b.write(folder/'global_FIT.json',r)
    for col,name in enumerate('BW'):
        ix=np.flatnonzero(m['colors'][ids]==col);r=p.fit_plda(x[ix],m['owners'][ids[ix]],folder/f'PLDA_{name}.npz',dimension=256);p.b.write(folder/f'{name}_FIT.json',r)
    bg=np.full(3*cleanbasis.BINS,1e-3,np.float64)
    for st in range(0,len(ids),2048):
        ix=ids[st:st+2048];codes=p.b.sequence_export(ix,m)
        for row in np.flatnonzero(m['colors'][ix]==1):valid=codes[row]!=65535;codes[row,valid]=(codes[row,valid]+362)%724
        bg+=np.asarray(cleanbasis.design(codes,'joint_order_D4').sum(0)).ravel()
    for g in range(3):bg[g*cleanbasis.BINS:(g+1)*cleanbasis.BINS]/=bg[g*cleanbasis.BINS:(g+1)*cleanbasis.BINS].sum()
    np.save(folder/'opening_background.npy',bg,allow_pickle=False);h8=np.load(dest/'H8_fit_indices.npy');cache=p.n.RUN/'feature_cache';receipt=p.b.read(cache/'RECEIPT.json');assert p.b.sha(cache/'vectors.npy')==receipt['vectors_sha256'];ci=np.load(cache/'indices.npy');lookup={int(i):j for j,i in enumerate(ci)};vec=np.load(cache/'vectors.npy',mmap_mode='r');raw=vec[[lookup[int(i)] for i in h8]];hf=folder/'H8';hf.mkdir();old.pre(raw,hf,True);rr=p.fit_plda(old.pre(raw,hf),m['owners'][h8],hf/'PLDA.npz',dimension=128);p.b.write(hf/'FIT.json',rr);p.b.write(dest/'MODELS_SHA256.json',{str(f):p.b.sha(f) for f in folder.rglob('*') if f.is_file()});p.b.log(stage='complete90_scale_cleanproxy_fit',histrows=len(ids),H8rows=len(h8))


def assemble():
    receipt=p.b.read(RUN/'dev_results/RECEIPT.json');assert receipt['all_valid'];ids=np.load(RUN/'all_indices.npy');oid=np.load(old.RUN/'dev_results/indices.npy');nid=np.load(RUN/'dev_results/indices.npy');assert np.array_equal(nid,np.load(RUN/'replay_indices.npy'));lookup={int(i):(0,j) for j,i in enumerate(oid)};lookup.update({int(i):(1,j) for j,i in enumerate(nid)});dest=RUN/'views';dest.mkdir(exist_ok=False)
    for v in (8,64,90):
        src=[np.load(root/f'view{v}.npy',mmap_mode='r') for root in (old.RUN/'dev_results',RUN/'dev_results')];a=np.lib.format.open_memmap(dest/f'view{v}.npy',mode='w+',dtype=np.float32,shape=(len(ids),2339))
        for j,i in enumerate(ids):k,z=lookup[int(i)];a[j]=src[k][z]
        a.flush()
    p.b.write(RUN/'ASSEMBLY_RECEIPT.json',{'indices':len(ids),'sameworkerSHA':True,'cached_train':8192,'new_replay':len(nid),'SHA256':{f.name:p.b.sha(f) for f in dest.glob('*.npy')}})


def train():
    assert p.b.sha(__file__)==p.b.read(RUN/'PROTOCOL.json')['source_SHA256'];ids=np.load(RUN/'all_indices.npy');pos={int(i):j for j,i in enumerate(ids)};dest=RUN/'models';dest.mkdir(exist_ok=False)
    for bank in BANKS:
        selected=np.load(RUN/f'train_{bank}.npy');labels=np.load(RUN/f'labels_{bank}.npy');at=[pos[int(i)] for i in selected]
        for view in (64,90):
            raw=np.load(RUN/f'views/view{view}.npy',mmap_mode='r')[at]
            for dim in (256,768):
                folder=dest/f'{bank}_raw{dim}_v{view}';folder.mkdir();x=old.pre(raw,folder,True);r=p.fit_plda(x,labels,folder/'PLDA.npz',dimension=dim);p.b.write(folder/'FIT_RECEIPT.json',dict(r,bank=bank,view=view,training_only=True));p.b.log(stage='complete90_scale_fit',bank=bank,view=view,**r)
    p.b.write(RUN/'MODELS_SHA256.json',{str(f):p.b.sha(f) for f in dest.rglob('*') if f.is_file()})


def sequence(codes,colors,ss,qs):
    codes=codes.copy();B=cleanbasis.BINS
    for row in np.flatnonzero(colors==1):valid=codes[row]!=65535;codes[row,valid]=(codes[row,valid]+362)%724
    x=cleanbasis.design(codes,'joint_order_D4');bg=np.load(RUN/'clean_proxy/models/opening_background.npy');bank=sparse.vstack([sparse.csr_matrix(x[sorted(s)].sum(0)) for s in ss],format='csr',dtype=np.float64);norm=np.stack([np.asarray(bank[:,g*B:(g+1)*B].sum(1)).ravel() for g in range(3)],1);bank.data=np.log1p(bank.data/(20*bg[bank.indices]));used=sorted({i for s in qs for i in s});qx=x[used];mass=np.stack([np.asarray(qx[:,g*B:(g+1)*B].sum(1)).ravel() for g in range(3)],1);value=(qx@bank.T).toarray()-mass@np.log1p(norm/20).T;pos={i:j for j,i in enumerate(used)};return np.stack([value[[pos[i] for i in sorted(s)]].mean(0)/3 for s in qs])


def baseline_and_components(ep,banks=BANKS):
    ids=np.load(RUN/'cal_indices.npy');allids=np.load(RUN/'all_indices.npy');at=np.searchsorted(allids,ids);assert np.array_equal(allids[at],ids);raw={v:np.load(RUN/f'views/view{v}.npy',mmap_mode='r')[at] for v in (8,64,90)};m=p.b.meta();pos={int(i):j for j,i in enumerate(ids)};ss=[[pos[i] for i in s] for s in ep['supports']];qs=[[pos[i] for i in z['indices']] for z in ep['questions']]
    features=np.asarray(np.load(p.ROOT/'data/processed/features.npy',mmap_mode='r')[m['rows'][ids]],np.float32);hist=t.cp.histograms(features,p.b.read(p.ROOT/'data/processed/feature_schema.json'),sqrt=True);gm=p.PLDA(RUN/'clean_proxy/models/global_PLDA.npz');z=gm.transform(hist);sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);gs=gm.score(qm,qn,sm,sn);cs=t.cp.color_score(features,m['colors'][ids],ss,qs,RUN/'clean_proxy/models');seq=sequence(p.b.sequence_export(ids,m),m['colors'][ids],ss,qs);fc=t.cp.fixed_rrf60(t.cp.fixed_rrf60(gs,cs),seq);hf=RUN/'clean_proxy/models/H8';hm=p.PLDA(hf/'PLDA.npz');z=hm.transform(old.pre(raw[8],hf));sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);hs=hm.score(qm,qn,sm,sn);prior=.75*fc+.25/(60+p.b.ranks(hs))
    g=old.component(old.PRIOR/'models/raw256_v64',raw[64],ids,ep);g768=cap.component(cap.RUN/'models/raw768',raw[64],ids,ep);old64=.5/(60+p.b.ranks(g))+.5/(60+p.b.ranks(g768));old90=.5/(60+p.b.ranks(old.component(old.RUN/'models/raw256_v90',raw[90],ids,ep)))+.5/(60+p.b.ranks(old.component(old.RUN/'models/raw768_v90',raw[90],ids,ep)));w=cap.countweights(ep);base=(1-w)*prior+w*(.5*old64+.5*old90);components={}
    for bank in banks:
        for view in (64,90):
            parts=[old.component(RUN/f'models/{bank}_raw{dim}_v{view}',raw[view],ids,ep) for dim in (256,768)];components[f'{bank}_v{view}']=.5/(60+p.b.ranks(parts[0]))+.5/(60+p.b.ranks(parts[1]))
    return base,old90,components,prior


def blend(base,old90,new90,ep,alpha,route):
    out=base.copy();mask=np.asarray([route=='all' or z['q']==1 for z in ep['questions']]);v=base+cap.countweights(ep)*.5*alpha*(new90-old90);out[mask]=v[mask];return out


def screen():
    ep=p.b.read(RUN/'EPISODE_screen.json');base,old90,c,prior=baseline_and_components(ep);dest=RUN/'screen';dest.mkdir(exist_ok=False)
    for k,v in dict(c,baseline=base,old90=old90,prior=prior).items():np.save(dest/f'{k}.npy',v,allow_pickle=False)
    results={};passed=[]
    for bank in BANKS:
        for alpha in (.5,1.):
            for route in ('Q1','all'):
                value=blend(base,old90,c[f'{bank}_v90'],ep,alpha,route);r=cap.compare(value,base,ep,route);paired=cap.compare(c[f'{bank}_v90'],c[f'{bank}_v64'],ep,route);r['checks']['paired_nonnegative']=paired['delta']>=0;r['passed']=all(r['checks'].values());r.update(bank=bank,alpha=alpha,route=route,paired90vs64=paired);key=f'{bank}_a{alpha:g}_{route}';results[key]=r
                if r['passed']:passed.append(key)
    winner=sorted(passed,key=lambda k:(-results[k]['macro_score_all'],1 if results[k]['route']=='Q1' else 4,results[k]['alpha'],results[k]['bank']))[0] if passed else None;p.b.write(RUN/'SCREEN_RESULTS.json',results);p.b.write(RUN/'WINNER_FREEZE.json',{'winner':winner,'selected':results[winner] if winner else None,'passing':passed,'confirmation_quality_unread':True});p.b.log(stage='complete90_scale_screen',winner=winner,passing=passed,summary={k:{'delta':v['delta'],'CI95':v['CI95'],'netTop1':v['Top1_net'],'byq':v['by_games_delta'],'paired':v['paired90vs64']['delta'],'passed':v['passed']} for k,v in results.items()})


def confirm():
    selected=p.b.read(RUN/'WINNER_FREEZE.json')['selected'];assert selected;ep=p.b.read(RUN/'EPISODE_confirmation.json');bank=selected['bank'];base,old90,c,prior=baseline_and_components(ep,(bank,));value=blend(base,old90,c[f'{bank}_v90'],ep,selected['alpha'],selected['route']);r=cap.compare(value,base,ep,selected['route']);paired=cap.compare(c[f'{bank}_v90'],c[f'{bank}_v64'],ep,selected['route']);r['checks']['paired_nonnegative']=paired['delta']>=0;r['passed']=all(r['checks'].values());r.update(bank=bank,alpha=selected['alpha'],route=selected['route'],paired90vs64=paired,new600_recentdisjoint=True,historical_pristine_not_proven=True,cleanproxy_notexactproduction=True);dest=RUN/'confirmation';dest.mkdir(exist_ok=False)
    for k,v in dict(c,baseline=base,old90=old90,prior=prior,candidate=value).items():np.save(dest/f'{k}.npy',v,allow_pickle=False)
    p.b.write(RUN/'CONFIRMATION_RESULT.json',r);p.b.log(stage='complete90_scale_confirmation',**r)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','proxy','assemble','train','screen','confirm']);v=parser.parse_args()
    with threadpool_limits(limits=4):{'prepare':prepare,'proxy':build_proxy,'assemble':assemble,'train':train,'screen':screen,'confirm':confirm}[v.stage]()
