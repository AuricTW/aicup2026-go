"""Frozen64 model extension to2/3/5games; new query banks, no fitting/upload."""
from pathlib import Path
import sys,argparse,time,hashlib
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parent))
import train_fresh600_dense64_20261111_v3 as t
import numpy as np
from threadpoolctl import threadpool_limits
p=t.p;q=t.q;PRIOR=q.RUN
RUN=p.ROOT/'runs/multi_dense64_extension_20261111_v1';SEED=2026111181
PARENT=p.ROOT/'submissions/20261111_v2_Q1_fresh600_raw256_64_replace_w1_v1/submission_player.csv'
ROUTES={'Q2':[2],'Q23':[2,3],'Q235':[2,3,5]}

def prepare():
    RUN.mkdir(exist_ok=False);assert p.b.sha(PARENT)=='7b2774967016006d86868e715c5e0d5ac13c85279e3ecf72c05398ee2d3d66e0';m=p.b.meta();prior=p.b.read(PRIOR/'EPISODE_screen.json');candidates=prior['candidates'];support=prior['supports']
    paths=[PRIOR/'cal_indices.npy',p.RUN/'cal_indices.npy',p.DENSE.parent/'validation_indices.npy',p.ROOT/'runs/v2_q1_metric_20261111_v1/fresh_indices.npy',p.ROOT/'runs/v2_q1_fixed_replication1200_20261111_v1/fresh_indices.npy'];old=np.unique(np.concatenate([np.load(f) for f in paths]));blocked={f:set(m[f][old]) for f in p.b.FAMILIES};by=defaultdict(list)
    for i in np.flatnonzero(np.isin(m['owners'].astype(str),candidates)&(m['hd']<=1)):
        if int(m['group'][i][:8],16)%10>=7 and all(m[f][i] not in blocked[f] for f in p.b.FAMILIES):by[m['owners'][i].decode()].append(int(i))
    eligible={owner:sorted(ids,key=lambda i:hashlib.sha256(m['group'][i]+str(SEED).encode()).digest()) for owner,ids in by.items() if len(ids)>=5};groups=defaultdict(list)
    for owner,ids in eligible.items():groups[str(m['ranks'][ids[0]])].append(owner)
    for rank in groups:groups[rank].sort(key=lambda x:hashlib.sha256((str(SEED)+x).encode()).digest())
    order=[];cursor=0
    while any(cursor<len(ids) for ids in groups.values()):
        for rank in sorted(groups):
            if cursor<len(groups[rank]):order.append(groups[rank][cursor])
        cursor+=1
    used={f:set() for f in p.b.FAMILIES};banks={}
    for owner in order:
        picked=[];local={f:set() for f in p.b.FAMILIES}
        for i in eligible[owner]:
            if any(m[f][i] in used[f] or m[f][i] in local[f] for f in p.b.FAMILIES):continue
            picked.append(i)
            for f in p.b.FAMILIES:local[f].add(m[f][i])
            if len(picked)==5:break
        if len(picked)!=5:continue
        banks[owner]=sorted(picked,key=lambda i:m['raw'][i])
        for f in p.b.FAMILIES:used[f].update(local[f])
        if len(banks)==400:break
    assert len(banks)==400,('insufficientfresh5gamebanks',len(banks),len(eligible))
    owners=sorted(banks,key=lambda x:hashlib.sha256(('split|'+str(SEED)+x).encode()).digest());eps=[]
    for part,players in [('screen',owners[:200]),('confirmation',owners[200:])]:
        questions=[]
        for count in (2,3,5):
            for owner in players:questions.append({'id':f'{part}_{count}_{len(questions):04}','indices':banks[owner][:count],'q':count,'target':candidates.index(owner),'player':owner})
        ep={'candidates':candidates,'supports':support,'questions':questions,'part':part,'seed':SEED,'nested2_3_5_query_sets':True};eps.append(ep);p.b.write(RUN/f'EPISODE_{part}.json',ep)
    fresh=np.asarray(sorted(i for b in banks.values() for i in b),np.int64);cal=np.asarray(sorted(set(np.concatenate(support).tolist())|set(fresh.tolist())),np.int64);assert len(fresh)==2000 and len(cal)==8000
    for name,a in [('replay_indices',fresh),('cal_indices',cal)]:np.save(RUN/f'{name}.npy',a,allow_pickle=False)
    for name in ['train_indices.npy','clean_proxy/fit_indices.npy','clean_proxy/H8_fit_indices.npy']:
        fit=np.load(PRIOR/name)
        for f in p.b.FAMILIES:assert not set(m[f][fit])&set(m[f][cal]),(name,f)
    pass  # leaderboard record removed for the public release
    p.b.write(RUN/'PROTOCOL.json',{'created_utc':p.b.base.utc(),'source_SHA256':p.b.sha(__file__),'metadata_seed':SEED,'bootstrap_seed':q.SEED,'hypothesis':'officiallyhelpful64decision representation mayalso improve2/3/5game identification',
      'all_models_and_weights_frozen':True,'no_training_or_refit':True,'new_query_unique_games':2000,'new_phase_screen_players':200,'new_phase_confirm_players':200,'queries_per_phase':600,'queries_per_length':200,'candidate_count':600,'support_games':10,
      'nested_queries':'2/3/5 fromsamefresh5gamebank perplayer, correlated; bootstrapbyplayer notbyquestion','screen_confirm_target_overlap':0,
      'exposure':'same600identitypool alreadyopened forQ1; new query4familiesdisjoint namedpriorbanks, not globallypristine orindependentidentityconfirmation','fresh_bank_exclusion_sources':[str(f) for f in paths],
      'model':'frozenraw256_v64, sameprior512training; matchedraw256_v32control; no newfittedassets',
      'routes':ROUTES,'formula':'keepnewQ1exactbestG64; replaceonlyrouted2/3/5neural0.4term withfrozen64model, no weightsearch',
      'clean_proxy':'samefrozen cleanhist/H8/opening+matchedraw12832; no identity-exposedlegacyparent quality used',
      'screen_gates':{'paired_component_and_fusion_mean_delta_gte':.01,'paired_Top1_net_gte':6,'fusion_Top1_net_gte':8,'each_routed_length_delta_gte':0,'sharedplayer95CI_lower_gt':0},
      'selection':'highestmeanpassingfusion over600queries, tie fewer routedlengths thenname','confirmation':'ONEselectedroute only, samegates butbothTop1netmin6; no runnerups ornewbanks afterfailure',
      'if_pass':'export challenger, preservingold800+newQ1 andnonroutedrows; actualproductionfusiongainunconfirmed untilofficialfeedback','if_fail':'retainG64, no unqualifiedextensionCSV',
      'official_missing_features':'onlyselectednewquerylengths aftergates, reuse6000supportperpool andallpreviousnative64cache','no_Publiclabels_infit_orlookup':True,'original_sealed400_not_read':True,'no_upload':True})
    protected={str(PARENT):p.b.sha(PARENT)}
    for folder in [PRIOR/'models/raw128_v32',PRIOR/'models/raw256_v32',PRIOR/'models/raw256_v64',PRIOR/'clean_proxy/models']:
        protected.update({str(f):p.b.sha(f) for f in folder.rglob('*') if f.is_file()})
    protected[str(p.ROOT/'submissions/20261111_v2_dense32_raw32_w0.4_v1/submission_player.csv')]=p.b.sha(p.ROOT/'submissions/20261111_v2_dense32_raw32_w0.4_v1/submission_player.csv')
    p.b.write(RUN/'FROZEN_ASSETS_SHA256.json',protected);p.b.write(RUN/'SCOPE_AUDIT.json',{'eligible_fresh5game_players':len(eligible),'selected':400,'fresh_query_unique_games':2000,'all_new_query4families_disjoint_prior_named_banks':True,'all_fit_vsfullnewcal4family_overlap':0,'same_previously_exposed600identities':True,'no_newfit':True});p.b.log(stage='multi64_prepared',eligible=len(eligible),players=400,query_games=2000)

def assemble():
    receipt=p.b.read(RUN/'dev_results/RECEIPT.json');assert receipt['all_valid']
    for f,h in receipt['SHA256'].items():assert p.b.sha(RUN/'dev_results'/f)==h
    ids=np.load(RUN/'cal_indices.npy');newids=np.load(RUN/'dev_results/indices.npy');assert np.array_equal(newids,np.load(RUN/'replay_indices.npy'));oldids=np.load(PRIOR/'cal_indices.npy');lookup={int(i):('old',j) for j,i in enumerate(oldids)};lookup.update({int(i):('new',j) for j,i in enumerate(newids)});dest=RUN/'views';dest.mkdir(exist_ok=False)
    for view in (8,32,64):
        sources={'old':np.load(PRIOR/('views/cal_original8.npy' if view==8 else f'views/cal{view}.npy'),mmap_mode='r'),'new':np.load(RUN/f'dev_results/view{view}.npy',mmap_mode='r')};a=np.stack([sources[lookup[int(i)][0]][lookup[int(i)][1]] for i in ids]);assert np.isfinite(a).all();np.save(dest/f'cal{view}.npy',a,allow_pickle=False)
    p.b.write(dest/'RECEIPT.json',{'games':len(ids),'support6000_reused_verified64cache':True,'new_queries2000':True,'no_fit':True})

def predict():
    for f,h in p.b.read(RUN/'FROZEN_ASSETS_SHA256.json').items():assert p.b.sha(f)==h
    ids=np.load(RUN/'cal_indices.npy');m=p.b.meta();eps=[p.b.read(RUN/f'EPISODE_{part}.json') for part in ('screen','confirmation')];pos={int(i):j for j,i in enumerate(ids)};raw=np.asarray(np.load(p.ROOT/'data/processed/features.npy',mmap_mode='r')[m['rows'][ids]],np.float32);hist=t.cp.histograms(raw,p.b.read(p.ROOT/'data/processed/feature_schema.json'),sqrt=True);codes=p.b.sequence_export(ids,m)
    globalmodel=p.PLDA(PRIOR/'clean_proxy/models/global_PLDA.npz');zg=globalmodel.transform(hist);model8=p.PLDA(PRIOR/'clean_proxy/models/H8/PLDA.npz');z8=model8.transform(q.oldtrain.pre(np.load(RUN/'views/cal8.npy'),PRIOR/'clean_proxy/models/H8'));model32=p.PLDA(PRIOR/'models/raw128_v32/PLDA.npz');z32=model32.transform(q.oldtrain.pre(np.load(RUN/'views/cal32.npy'),PRIOR/'models/raw128_v32'))
    pair32=t.component(PRIOR/'models/raw256_v32',np.load(RUN/'views/cal32.npy'),ids,eps);new64=t.component(PRIOR/'models/raw256_v64',np.load(RUN/'views/cal64.npy'),ids,eps);dest=RUN/'predictions';dest.mkdir(exist_ok=False)
    for j,(part,ep) in enumerate(zip(('screen','confirmation'),eps)):
        ss=[[pos[i] for i in s] for s in ep['supports']];qs=[[pos[i] for i in z['indices']] for z in ep['questions']];sm,sn=p.summarize(zg,ss);qm,qn=p.summarize(zg,qs);g=globalmodel.score(qm,qn,sm,sn);col=t.cp.color_score(raw,m['colors'][ids],ss,qs,PRIOR/'clean_proxy/models');seq=t.clean.sequence_scores(codes,m['colors'][ids],ss,qs);fc=t.cp.fixed_rrf60(t.cp.fixed_rrf60(g,col),seq);sm,sn=p.summarize(z8,ss);qm,qn=p.summarize(z8,qs);k=model8.score(qm,qn,sm,sn);sm,sn=p.summarize(z32,ss);qm,qn=p.summarize(z32,qs);r32=model32.score(qm,qn,sm,sn);prior=.75*fc+.25/(60+p.b.ranks(k));base=.6*prior+.4/(60+p.b.ranks(r32));full=.6*prior+.4/(60+p.b.ranks(new64[j]));dd=dest/part;dd.mkdir()
        for name,a in [('baseline',base),('replacement',full),('pair32',pair32[j]),('new64',new64[j])]:np.save(dd/f'{name}.npy',a,allow_pickle=False)
    p.b.write(dest/'RECEIPT.json',{'all_models_frozen':True,'quality_not_calculated':True,'cleanproxy_not_exactproductionparent':True,'no_fit_or_official_labels':True})

def compare(a,b,ep,route,minnet):
    report=t.compare(a,b,ep,minnet=minnet);av,_=t.values(a,ep);bv,_=t.values(b,ep);delta=av-bv;by={str(n):float(delta[np.asarray([z['q']==n for z in ep['questions']])].mean()) for n in (2,3,5)};report['by_games_delta']=by;report['checks']['each_routed_length']=all(by[str(n)]>=0 for n in ROUTES[route]);report['passed']=all(report['checks'].values());return report

def route_result(part,route,confirm=False):
    ep=p.b.read(RUN/f'EPISODE_{part}.json');dest=RUN/f'predictions/{part}';mask=np.asarray([z['q'] in ROUTES[route] for z in ep['questions']]);base=np.load(dest/'baseline.npy');score=base.copy();score[mask]=np.load(dest/'replacement.npy')[mask];control=np.load(dest/'pair32.npy');paired=control.copy();paired[mask]=np.load(dest/'new64.npy')[mask];pr=compare(paired,control,ep,route,6);fr=compare(score,base,ep,route,6 if confirm else 8);return {'route':route,'counts':ROUTES[route],'paired64_vs32':pr,'fusion_vs_cleanproxy':fr,'passed':pr['passed'] and fr['passed']},score

def screen():
    results={};passing=[]
    for route in ROUTES:
        r,_=route_result('screen',route);results[route]=r
        if r['passed']:passing.append(route)
    best=sorted(passing,key=lambda k:(-results[k]['fusion_vs_cleanproxy']['score'],len(ROUTES[k]),k))[0] if passing else None
    p.b.write(RUN/'SCREEN_RESULTS.json',results);p.b.write(RUN/'WINNER_FREEZE.json',{'winner':best,'counts':ROUTES[best] if best else [],'selected':results[best] if best else None,'passing':passing,'new_multigame_confirmation_outcomes_unread':True});p.b.log(stage='multi64_screen',winner=best,results=results)

def confirm():
    freeze=p.b.read(RUN/'WINNER_FREEZE.json');assert freeze['winner'];r,score=route_result('confirmation',freeze['winner'],confirm=True);r.update({'models_unchanged':True,'same600identities_priorQ1exposed':True,'new_multigame_screen200_confirmation200_targetdisjoint':True,'nested_queries_shared_player_bootstrap':True,'clean_proxy_not_exactproduction':True});np.save(RUN/'confirmation_fused.npy',score,allow_pickle=False);p.b.write(RUN/'CONFIRMATION_RESULT.json',r);p.b.log(stage='multi64_confirmation',**r)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','assemble','predict','screen','confirm']);v=a.parse_args()
    with threadpool_limits(limits=4):{'prepare':prepare,'assemble':assemble,'predict':predict,'screen':screen,'confirm':confirm}[v.stage]()
