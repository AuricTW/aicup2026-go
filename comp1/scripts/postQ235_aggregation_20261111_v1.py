"""Bounded frozen64 aggregation pilot; prospective fresh-game confirmation."""
from pathlib import Path
import sys,argparse,hashlib
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parent))
import multi_dense64_extension_20261111_v1 as old
import numpy as np
from threadpoolctl import threadpool_limits
p=old.p;t=old.t;q=old.q;PRIOR=old.PRIOR
RUN=p.ROOT/'runs/postQ235_aggregation_20261111_v1';SEED=2026111191
PARENT=p.ROOT/'submissions/20261111_v2_dense64_Q235_extension_v1/submission_player.csv'
SPECS=[{'kind':'color','tau':tau,'alpha':a,'route':route} for tau in (0,2) for a in (.5,1.) for route in ('Q1','all')]+[{'kind':'weight','weight':w,'route':route} for w in (.5,.6,.7) for route in ('Q235','all')]


def prepare():
    RUN.mkdir(exist_ok=False);assert p.b.sha(PARENT)=='e64eb9d5aa03586c8f1a7eb6c18bf071fb3cf2306504b764a8376392d73661bd'
    m=p.b.meta();a=p.b.read(PRIOR/'EPISODE_screen.json');b=p.b.read(old.RUN/'EPISODE_screen.json')
    chosen=sorted({z['player'] for z in a['questions']},key=lambda x:hashlib.sha256((str(SEED)+x).encode()).digest())[:200]
    by={}
    for z in a['questions']:
        if z['player'] in chosen and z['player'] not in by:by[z['player']]=dict(z,source='Q1',source_index=a['questions'].index(z))
    questions=[by[x] for x in chosen]+[dict(z,source='multi',source_index=i) for i,z in enumerate(b['questions'])]
    screen=dict(a,questions=questions,part='screen',seed=SEED);p.b.write(RUN/'EPISODE_screen.json',screen)
    screenplayers={z['player'] for z in questions};assert len(questions)==800
    paths=[PRIOR/'cal_indices.npy',old.RUN/'cal_indices.npy',p.RUN/'cal_indices.npy',p.DENSE.parent/'validation_indices.npy',p.ROOT/'runs/v2_q1_metric_20261111_v1/fresh_indices.npy',p.ROOT/'runs/v2_q1_fixed_replication1200_20261111_v1/fresh_indices.npy']
    priorids=np.unique(np.concatenate([np.load(f) for f in paths]));blocked={f:set(m[f][priorids]) for f in p.b.FAMILIES};pool=defaultdict(list)
    for i in np.flatnonzero(np.isin(m['owners'].astype(str),a['candidates']) & (m['hd']<=1)):
        owner=m['owners'][i].decode()
        if owner not in screenplayers and all(m[f][i] not in blocked[f] for f in p.b.FAMILIES):pool[owner].append(int(i))
    groups=defaultdict(list)
    for owner,ids in pool.items():
        if len(ids)>=5:groups[str(m['ranks'][ids[0]])].append(owner)
    for rank in groups:groups[rank].sort(key=lambda x:hashlib.sha256(('confirm|'+str(SEED)+x).encode()).digest())
    ordered=[];cursor=0
    while any(cursor<len(v) for v in groups.values()):
        for rank in sorted(groups):
            if cursor<len(groups[rank]):ordered.append(groups[rank][cursor])
        cursor+=1
    banks={};used={f:set() for f in p.b.FAMILIES}
    for owner in ordered:
        picked=[];local={f:set() for f in p.b.FAMILIES}
        for i in sorted(pool[owner],key=lambda i:hashlib.sha256(m['group'][i]+str(SEED).encode()).digest()):
            if any(m[f][i] in used[f] or m[f][i] in local[f] for f in p.b.FAMILIES):continue
            picked.append(i)
            for f in p.b.FAMILIES:local[f].add(m[f][i])
            if len(picked)==5:break
        if len(picked)<5:continue
        banks[owner]=sorted(picked,key=lambda i:m['raw'][i])
        for f in p.b.FAMILIES:used[f].update(local[f])
        if len(banks)==200:break
    assert len(banks)==200,('fresh_target_bank_capacity',len(banks),len(screenplayers),len(ordered))
    confirmation=dict(a,questions=[{'id':f'new_{n}_{j:04}','player':owner,'target':a['candidates'].index(owner),'q':n,'indices':banks[owner][:n]} for n in (1,2,3,5) for j,owner in enumerate(sorted(banks))],part='confirmation',seed=SEED)
    p.b.write(RUN/'EPISODE_confirmation.json',confirmation)
    fresh=np.asarray(sorted(i for bank in banks.values() for i in bank),np.int64);cal=np.asarray(sorted(set(np.concatenate(a['supports']).tolist())|set(fresh.tolist())),np.int64)
    assert len(fresh)==1000 and len(cal)==7000 and not screenplayers&set(banks)
    for name,arr in [('replay_indices',fresh),('cal_indices',cal)]:np.save(RUN/f'{name}.npy',arr,allow_pickle=False)
    for name in ('train_indices.npy','clean_proxy/fit_indices.npy','clean_proxy/H8_fit_indices.npy'):
        fit=np.load(PRIOR/name)
        assert not set(m['owners'][fit])&set(m['owners'][cal])
        for f in p.b.FAMILIES:assert not set(m[f][fit])&set(m[f][cal])
    feedback={};p.b.write(RUN/'OFFICIAL_FEEDBACK.json',feedback)
    protected=dict(p.b.read(old.RUN/'FROZEN_ASSETS_SHA256.json'));protected[str(PARENT)]=p.b.sha(PARENT)
    for root in (PRIOR,old.RUN):
        manifest=p.b.read(root/'FINAL_MANIFEST_SHA256.json')
        for f in root.rglob('*.npy'):
            if ('cal' in f.name or f.parent.name in ('views','parent','screen','official_predictions') or 'predictions' in f.parts) and str(f.relative_to(p.ROOT)) in manifest:protected[str(f)]=manifest[str(f.relative_to(p.ROOT))]
    p.b.write(RUN/'PROTECTED_INPUTS_SHA256.json',protected)
    p.b.write(RUN/'PROTOCOL.json',{'created_utc':p.b.base.utc(),'source_SHA256':p.b.sha(__file__),'seed':SEED,'specs':SPECS,'configs':len(SPECS),'baseline':'frozen cleanproxy with64 global neural0.4; deployedQ235 remains rollback','hypotheses':['color-conditioned support and query may remove B/W style mismatch','successful64 component may merit more weight than inherited0.4'],
       'color':'same frozen raw256_v64 PLDA; split query by actual targetcolor; supports samecolor or global fallback; tau2 shrink samecolor mean to global, effective count from exact pergame squaredweights; query color scores weighted by query count','color_fusion':'replace0.4 rankterm by (1-alpha)*globalRRF60+alpha*colorRRF60, no fit','weight_fusion':'(1-w)*frozenprior+w*RRF60(global64); no parameter fit',
       'screen':'200 Q1(one each chosen from oldscreen only)+600 multi(200 each2/3/5) =800; earlier screen evidence reused, not fresh','confirmation':'new1000uniquegames;200 targetplayers disjoint thisscreen union;800nested1/2/3/5queries; same previously exposed600pool; not globallynew identities','gate_both_stages':{'SID_delta_min':.0075,'Top1_net_min':6,'player_cluster95CI_lower_gt':0,'each_routed_length_delta_min':0},'selection':'highestpassingmean, tie fewer routedlengths thenkey; onlyONEwinner confirmation; no runnerup afterfailure','bootstrap':'5000sharedplayer clusters, not800independentplayers',
       'query_games_exclude':list(map(str,paths)),'all_newquery4families_disjoint_namedpriorbanks':True,'all_frozenfit_full600owner_and4family_excluded':True,'support10_candidates600':True,'no_training_or_refit':True,'no_Publicanswers_sealed400_or_upload':True,'if_pass':'export challenger, old800exactQ235; existingmodels preserved','if_fail':'retain failure, no candidate fromthisbranch','GPU':'onlyfresh1000feature replay on4090 after idle+ownership check, exactpriorworker; cap5400s; no environment change'})
    p.b.write(RUN/'SCOPE_AUDIT.json',{'screen_uniqueplayers':len(screenplayers),'confirmation_players':len(banks),'screen_confirm_target_overlap':0,'new_uniquegames':1000,'all_fit_owner_family_overlap':0,'all_queries_fourfamily_disjoint_namedpriorbanks':True,'safe_metadata_SHA256':p.b.sha(p.b.SAFE),'original_sealed400_not_opened':True,'same_exposed600':True});p.b.log(stage='postQ235_prepared',screenplayers=len(screenplayers),confirmplayers=len(banks),configs=len(SPECS))


def color_component(raw,ids,colors,ep,tau):
    order=np.argsort(ids,kind='stable');ids=ids[order];raw=raw[order];colors=colors[order];folder=PRIOR/'models/raw256_v64';model=p.PLDA(folder/'PLDA.npz');z=model.transform(t.prepare_x(raw,folder));pos={int(i):j for j,i in enumerate(ids)}
    ss=[[pos[i] for i in s] for s in ep['supports']];qs=[[pos[i] for i in v['indices']] for v in ep['questions']];globalmean,globaln=p.summarize(z,ss);score=np.zeros((len(qs),len(ss)),np.float64)
    for c in (0,1):
        cm=[];cn=[]
        for s,g,nall in zip(ss,globalmean,globaln):
            same=[i for i in sorted(s) if colors[i]==c];n=len(same)
            if n==0:cm.append(g);cn.append(float(nall));continue
            a=n/(n+tau);cm.append(a*z[same].mean(0)+(1-a)*g)
            weights=np.asarray([a/n+(1-a)/nall if colors[i]==c else (1-a)/nall for i in sorted(s)]);cn.append(float(1/(weights@weights)))
        subset=[[i for i in v if colors[i]==c] for v in qs];at=[i for i,s in enumerate(subset) if s]
        if at:
            qm,qn=p.summarize(z,[subset[i] for i in at]);v=model.score(qm,qn,np.stack(cm),np.asarray(cn))
            for j,i in enumerate(at):score[i]+=v[j]*(len(subset[i])/len(qs[i]))
    return score


def routecounts(spec):return {'Q1':(1,),'Q235':(2,3,5),'all':(1,2,3,5)}[spec['route']]
def key(spec):return '_'.join(f'{k}{v}' for k,v in spec.items())
def fuse(base,prior,global64,col,spec,ep):
    if spec['kind']=='color':changed=.6*prior+.4*((1-spec['alpha'])/(60+p.b.ranks(global64))+spec['alpha']/(60+p.b.ranks(col)))
    else:changed=(1-spec['weight'])*prior+spec['weight']/(60+p.b.ranks(global64))
    take=np.asarray([z['q'] in routecounts(spec) for z in ep['questions']]);score=base.copy();score[take]=changed[take];return score
def compare(score,base,ep,spec):
    r=t.compare(score,base,ep,minmean=.0075,minnet=6);av,_=t.values(score,ep);bv,_=t.values(base,ep);delta=av-bv;by={str(n):float(delta[np.asarray([z['q']==n for z in ep['questions']])].mean()) for n in (1,2,3,5)};r['by_games_delta']=by;r['checks']['each_routed_length']=all(by[str(n)]>=0 for n in routecounts(spec));r['passed']=all(r['checks'].values());return r


def screen():
    assert p.b.sha(__file__)==p.b.read(RUN/'PROTOCOL.json')['source_SHA256'];m=p.b.meta();ep=p.b.read(RUN/'EPISODE_screen.json');priorparts=[];globalparts=[];colorparts={tau:[] for tau in (0,2)}
    for source,root in [('Q1',PRIOR),('multi',old.RUN)]:
        at=[z['source_index'] for z in ep['questions'] if z['source']==source];sourceep=p.b.read(root/'EPISODE_screen.json');sourceep=dict(sourceep,questions=[sourceep['questions'][i] for i in at]);ids=np.load(root/'cal_indices.npy');raw=np.load(root/'views/cal64.npy',mmap_mode='r');g=t.component(PRIOR/'models/raw256_v64',raw,ids,[sourceep])[0];globalparts.append(g)
        if source=='Q1':priorparts.append(np.load(root/'parent/prior_screen.npy')[at])
        else:
            base=np.load(root/'predictions/screen/replacement.npy')[at];gg=np.load(root/'predictions/screen/new64.npy')[at];assert np.array_equal(g,gg);priorparts.append((base-.4/(60+p.b.ranks(gg)))/.6)
        for tau in (0,2):colorparts[tau].append(color_component(raw,ids,m['colors'][ids],sourceep,tau))
    g=np.concatenate(globalparts);prior=np.concatenate(priorparts);col={tau:np.concatenate(v) for tau,v in colorparts.items()};base=.6*prior+.4/(60+p.b.ranks(g));dest=RUN/'screen';dest.mkdir(exist_ok=False)
    for name,a in [('global64',g),('prior',prior),('baseline',base)]+[(f'color_tau{tau}',a) for tau,a in col.items()]:np.save(dest/f'{name}.npy',a,allow_pickle=False)
    results={};passing=[]
    for spec in SPECS:
        k=key(spec);score=fuse(base,prior,g,col.get(spec.get('tau')),spec,ep);r=compare(score,base,ep,spec);r['spec']=spec;results[k]=r
        if r['passed']:passing.append(k)
    winner=sorted(passing,key=lambda k:(-results[k]['score'],len(routecounts(results[k]['spec'])),k))[0] if passing else None
    p.b.write(RUN/'SCREEN_RESULTS.json',results);p.b.write(RUN/'WINNER_FREEZE.json',{'winner':winner,'selected':results[winner] if winner else None,'passing':passing,'new_confirmation_quality_unread':True});p.b.log(stage='postQ235_screen',winner=winner,passing=passing,results={k:{'delta':v['delta'],'Top1net':v['Top1_net'],'byq':v['by_games_delta'],'CI95':v['CI95']} for k,v in results.items()})


def assemble():
    root=RUN/'dev_results';receipt=p.b.read(root/'RECEIPT.json');assert receipt['all_valid'] and not receipt['fit_or_optimizer_used']
    for file,h in receipt['SHA256'].items():assert p.b.sha(root/file)==h
    oldids=np.load(PRIOR/'cal_indices.npy');newids=np.load(root/'indices.npy');assert np.array_equal(newids,np.load(RUN/'replay_indices.npy'));ids=np.load(RUN/'cal_indices.npy');pos={int(i):('old',j) for j,i in enumerate(oldids)};pos.update({int(i):('new',j) for j,i in enumerate(newids)});dest=RUN/'views';dest.mkdir(exist_ok=False)
    for v in (8,32,64):
        data={'old':np.load(PRIOR/('views/cal_original8.npy' if v==8 else f'views/cal{v}.npy'),mmap_mode='r'),'new':np.load(root/f'view{v}.npy',mmap_mode='r')};a=np.stack([data[pos[int(i)][0]][pos[int(i)][1]] for i in ids]);np.save(dest/f'cal{v}.npy',a,allow_pickle=False)


def confirm():
    freeze=p.b.read(RUN/'WINNER_FREEZE.json');assert freeze['winner'];spec=freeze['selected']['spec'];ep=p.b.read(RUN/'EPISODE_confirmation.json');ids=np.load(RUN/'cal_indices.npy');m=p.b.meta();pos={int(i):j for j,i in enumerate(ids)};ss=[[pos[i] for i in s] for s in ep['supports']];qs=[[pos[i] for i in z['indices']] for z in ep['questions']]
    raw=np.asarray(np.load(p.ROOT/'data/processed/features.npy',mmap_mode='r')[m['rows'][ids]],np.float32);hist=t.cp.histograms(raw,p.b.read(p.ROOT/'data/processed/feature_schema.json'),sqrt=True);codes=p.b.sequence_export(ids,m);gm=p.PLDA(PRIOR/'clean_proxy/models/global_PLDA.npz');z=gm.transform(hist);sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);gstat=gm.score(qm,qn,sm,sn);cstat=t.cp.color_score(raw,m['colors'][ids],ss,qs,PRIOR/'clean_proxy/models');seq=t.clean.sequence_scores(codes,m['colors'][ids],ss,qs);fc=t.cp.fixed_rrf60(t.cp.fixed_rrf60(gstat,cstat),seq)
    hm=p.PLDA(PRIOR/'clean_proxy/models/H8/PLDA.npz');z=hm.transform(q.oldtrain.pre(np.load(RUN/'views/cal8.npy'),PRIOR/'clean_proxy/models/H8'));sm,sn=p.summarize(z,ss);qm,qn=p.summarize(z,qs);h=hm.score(qm,qn,sm,sn);prior=.75*fc+.25/(60+p.b.ranks(h));raw64=np.load(RUN/'views/cal64.npy',mmap_mode='r');g=t.component(PRIOR/'models/raw256_v64',raw64,ids,[ep])[0];c=color_component(raw64,ids,m['colors'][ids],ep,spec['tau']) if spec['kind']=='color' else None;base=.6*prior+.4/(60+p.b.ranks(g));score=fuse(base,prior,g,c,spec,ep);r=compare(score,base,ep,spec);r.update({'winner':freeze['winner'],'spec':spec,'same_exposed600_not_pristine_identity':True,'new1000games_vs_namedbanks':True,'target_disjoint_thisscreen':True,'clean_proxy_not_exactproduction':True});dest=RUN/'confirmation';dest.mkdir(exist_ok=False)
    for name,a in [('baseline',base),('candidate',score),('global64',g),('prior',prior)]+([('color',c)] if c is not None else []):np.save(dest/f'{name}.npy',a,allow_pickle=False)
    p.b.write(RUN/'CONFIRMATION_RESULT.json',r);p.b.log(stage='postQ235_confirmation',**r)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','screen','assemble','confirm']);v=parser.parse_args()
    with threadpool_limits(limits=4):{'prepare':prepare,'screen':screen,'assemble':assemble,'confirm':confirm}[v.stage]()
