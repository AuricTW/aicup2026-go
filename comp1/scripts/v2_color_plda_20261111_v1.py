"""Prospective low-cost extension for the newly released mixed-color task.

No test labels. Original-safe data only, no retired identities. The existing
baseline programs/results stay immutable. New fitted metrics are separate.
"""
from pathlib import Path
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_v2_baseline_20261111_v2 as b
import numpy as np
from collections import Counter
from goid_v4.plda_metric_v1 import fit,PLDA,histograms,summarize
from goid_submission_v1.ensemble_D_v2 import fixed_rrf60
from threadpoolctl import threadpool_limits
RUN=b.RUN/'color_PLDA_pilot_v1'
FIT_BASE=b.ROOT/'runs/score_push_experiments_20261104_v2'

def prepare():
    RUN.mkdir(exist_ok=False)
    assert not (b.RUN/'calibration_human/SCORING_COMPLETE.json').exists()
    m=b.meta();ep=b.read(b.RUN/'calibration/ep0.json')
    blocked=set(ep['candidates'])|set(b.read(b.ROOT/'runs/4090_support100_pilot_20261105_v1/EPISODE_A.json')['candidates'])
    mask=np.isin(m['owners'].astype(str),sorted(blocked))
    fitids=np.load(FIT_BASE/'base_indices.npy');keep=(~mask[fitids])&(m['hd'][fitids]<=1)
    for f in b.FAMILIES:keep &= ~np.isin(m[f][fitids],np.unique(m[f][mask]))
    fitids=fitids[keep];assert len(fitids)>20000
    np.save(RUN/'fit_indices.npy',fitids,allow_pickle=False)
    b.write(RUN/'PROTOCOL.json',{'created_utc':b.base.utc(),'motivation':'old rawPLDA is B-only; new release includes target W and mixed-color sets',
      'fit_rows':len(fitids),'by_color':dict(Counter(map(int,m['colors'][fitids]))),'fit_players':len(set(m['owners'][fitids])),
      'excludes_all600_validation_candidates_and_current_A_B_owners_and_all_families':True,
      'source':b.sha(b.SAFE),'fit_ids_sha256':b.sha(RUN/'fit_indices.npy'),'retired400_excluded_by_safe_source':True,
      'feature':'same1594 raw features -> sqrt normalized1582 histograms','model':'one256-dimensional PLDA per target color; same original fitter/shrinkage',
      'set_scoring':'color-specific predictive PLDA, query-color frequency weighted; if a support color absent, original global support set fallback',
      'recipes':['E_C=RRF60(D,colorPLDA)','F_C=RRF60(E_C,sequence)'],
      'HumanSL_weights':[.1,.25,.5],'HumanSL_quality_not_yet_read':True,
      'new_component_gate_vs_frozen_F':{'mean_delta_gte':.003,'each_episode_delta_gte':0,'q1_delta_gte':0,'q2_delta_gte':0,'top1_net_gte':0,'player_cluster95_lower_gt':0,'draws':5000,'seed':2026111103},
      'original_frozen_candidates':'D,E,F,E_H,H; plus predefined .25/.5 HumanSL fusions and K-only diagnostic',
      'all_new_weight_or_component_candidates_must_pass_both_D_and_F_gates':True,
      'if_none':'choose original D/E/F/E_H/H by original frozen gate only',
      'refit':'only after a colorPLDA recipe passes; refit B/W metrics on original-safe train scope, still exclude all current_A_B owners/families; no retired400 or official inputs',
      'no_test_fitting_or_upload':True,'not_independent_validation':True})
    b.write(RUN/'SOURCE_FREEZE.json',{'script':str(Path(__file__)),'sha256':b.sha(__file__)})
    b.log(stage='colorPLDA_prepared',fit_rows=len(fitids))

def fit_models(final=False):
    m=b.meta()
    if final:
        assert b.read(b.RUN/'SELECTION_EXTENDED.json')['selected'].split('_H')[0] in ('E_C','F_C')
        blocked=set(b.read(b.ROOT/'runs/4090_support100_pilot_20261105_v1/EPISODE_A.json')['candidates'])
        mask=np.isin(m['owners'].astype(str),sorted(blocked));keep=~mask&(m['hd']<=1)
        for f in b.FAMILIES:keep &= ~np.isin(m[f],np.unique(m[f][mask]))
        ids=np.flatnonzero(keep);dest=RUN/'final_models'
    else:ids=np.load(RUN/'fit_indices.npy');dest=RUN/'cal_models'
    dest.mkdir(exist_ok=False);raw=np.load(b.ROOT/'data/processed/features.npy',mmap_mode='r');schema=b.read(b.ROOT/'data/processed/feature_schema.json');receipt={}
    for c,color in enumerate('BW'):
        ix=ids[m['colors'][ids]==c]
        x=np.lib.format.open_memmap(dest/f'fit_{color}.npy',mode='w+',dtype=np.float32,shape=(len(ix),1582))
        for st in range(0,len(ix),2048):x[st:st+2048]=histograms(raw[m['rows'][ix[st:st+2048]]],schema,sqrt=True)
        x.flush();report=fit(x,m['owners'][ix],dest/f'PLDA_{color}.npz',dimension=256)
        receipt[color]=report;b.log(stage='colorPLDA_fit',color=color,final=final,**report)
    b.write(dest/'FIT_RECEIPT.json',{'models':receipt,'rows':len(ids),'fit_players':len(set(m['owners'][ids])),'models_SHA256':{c:b.sha(dest/f'PLDA_{c}.npz') for c in 'BW'},'calibration_quality':None,'no_official_data_or_labels':True})

def color_score(raw,colors,ss,qs,models):
    x=histograms(raw,b.read(b.ROOT/'data/processed/feature_schema.json'),sqrt=True)
    result=np.zeros((len(qs),len(ss)),np.float64)
    for c,color in enumerate('BW'):
        model=PLDA(models/f'PLDA_{color}.npz');z=model.transform(x)
        sc=[[i for i in sorted(s) if colors[i]==c] for s in ss]
        sc=[s if s else sorted(ss[j]) for j,s in enumerate(sc)]
        qc=[[i for i in sorted(q) if colors[i]==c] for q in qs];which=[i for i,q in enumerate(qc) if q]
        sm,sn=summarize(z,sc);qm,qn=summarize(z,[qc[i] for i in which]);v=model.score(qm,qn,sm,sn)
        for j,i in enumerate(which):result[i]+=v[j]*(len(qc[i])/len(qs[i]))
    assert np.isfinite(result).all();return result

def cal_predict():
    m=b.meta();ix=np.load(b.RUN/'calibration/indices.npy');pos={int(i):j for j,i in enumerate(ix)}
    raw=np.array(np.load(b.ROOT/'data/processed/features.npy',mmap_mode='r')[m['rows'][ix]],np.float32)
    for n in range(2):
        ep=b.read(b.RUN/f'calibration/ep{n}.json');ss=[[pos[i] for i in s] for s in ep['supports']];qs=[[pos[i] for i in q['indices']] for q in ep['questions']]
        v=color_score(raw,m['colors'][ix],ss,qs,RUN/'cal_models');d=RUN/f'ep{n}';d.mkdir()
        e=fixed_rrf60(np.load(b.RUN/f'calibration/stats{n}/D.npy'),v)
        f=fixed_rrf60(e,np.load(b.RUN/f'calibration/stats{n}/sequence.npy'))
        for name,s in [('color_PLDA',v),('E_C',e),('F_C',f)]:np.save(d/f'{name}.npy',s,allow_pickle=False)
    b.log(stage='colorPLDA_calibration_predictions_ready')

def candidate_scores(n,official=False):
    part=('public2','private2')[n] if official else f'ep{n}'
    source=b.RUN/f'official_stat/{part}' if official else b.RUN/f'calibration/stats{n}'
    human=b.RUN/f'official_human/{part}' if official else b.RUN/f'calibration_human/ep{n}'
    color=RUN/f'official/{part}' if official else RUN/f'ep{n}'
    result={name:np.load(source/f'{name}.npy') for name in ('D','E','F')}
    result.update({name:np.load(color/f'{name}.npy') for name in ('E_C','F_C')})
    k=np.load(human/'K.npy');kr=1/(60+b.ranks(k))
    for prior in ('E','F','E_C','F_C'):
        for w in (.1,.25,.5):
            # For F/E keep historical names for the original .1 recipe.
            name=('H' if prior=='F' else 'E_H') if w==.1 and prior in ('E','F') else prior+'_H'+str(int(w*100))
            result[name]=(1-w)*result[prior]+w*kr
    return result

def selection():
    # Produce the original frozen selection first, then allow extensions only
    # with the additional predefined gate against F, preventing an extension
    # from inheriting F's improvement over a weaker comparator.
    b.select();original=b.read(b.RUN/'SELECTION.json')['selected']
    eps=[b.read(b.RUN/f'calibration/ep{n}.json') for n in range(2)];scores=[candidate_scores(n) for n in range(2)]
    names=sorted(scores[0]);values={};metrics={}
    for name in names:
        rr=[];vv=[];mm=[]
        for ep,s in zip(eps,scores):
            t=np.asarray([q['target'] for q in ep['questions']]);r=b.ranks(s[name])[np.arange(len(t)),t]
            rr.append(r);vv.append(np.where(r<=5,np.exp(-(r-1)),0.));mm.append(b.metrics(s[name],ep))
        values[name]=(rr,vv);metrics[name]=mm
    players=sorted({q['player'] for ep in eps for q in ep['questions']});pp={p:i for i,p in enumerate(players)}
    draws=np.random.default_rng(2026111103).integers(len(players),size=(5000,len(players)))
    comparisons={};passing=[]
    for name in names:
        if name in ('D','E','F','E_H','H'):continue
        comparisons[name]={}
        for comparator in ('D','F'):
            diff=[v-w for v,w in zip(values[name][1],values[comparator][1])];sums=np.zeros(len(players));counts=np.zeros(len(players));net=0
            for j,ep in enumerate(eps):
                for q,v in zip(ep['questions'],diff[j]):i=pp[q['player']];sums[i]+=v;counts[i]+=1
                net+=int((values[name][0][j]==1).sum()-(values[comparator][0][j]==1).sum())
            boot=sums[draws].sum(1)/counts[draws].sum(1);ci=np.quantile(boot,[.025,.975]).tolist()
            byq={q:float(np.concatenate([v[np.asarray([z['q']==q for z in ep['questions']])] for v,ep in zip(diff,eps)]).mean()) for q in (1,2,3,5)}
            delta=float(np.concatenate(diff).mean());passed=delta>=.003 and all(d.mean()>=0 for d in diff) and byq[1]>=0 and byq[2]>=0 and net>=0 and ci[0]>0
            comparisons[name][comparator]={'delta':delta,'CI95':ci,'by_games_delta':byq,'episode_delta':[float(d.mean()) for d in diff],'top1_net':net,'passed':bool(passed)}
        if all(c['passed'] for c in comparisons[name].values()):passing.append(name)
    eligible=[original]+passing
    selected=sorted(eligible,key=lambda n:(-np.mean([v['score'] for v in metrics[n]]),n))[0]
    b.write(b.RUN/'SELECTION_EXTENDED.json',{'selected':selected,'original_selection':original,'metrics':metrics,'comparisons':comparisons,'eligible':eligible,'not_independent_or_official':True,'Public_labels_loaded':False})
    b.log(stage='extended_selection',selected=selected,original=original,mean_scores={k:float(np.mean([v['score'] for v in mm])) for k,mm in metrics.items()})

def official_predict():
    games,pools=b.load_official();keys=b.read(b.RUN/'official_stat/keys.json');pos={k:i for i,k in enumerate(keys)}
    # Reuse validated replay-derived stats, not training or test label arrays.
    import sqlite3
    cache=b.RUN/'cache/stats.sqlite';db=sqlite3.connect(cache.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        rows={r[0]:r for r in db.execute('SELECT * FROM u0_v1')};assert set(rows)==set(keys)
        from goid_v4.csv_u0_v1 import Cache
        raw=np.stack([Cache.check(rows[k],games[k])[0] for k in keys])
    finally:db.close()
    colors=np.asarray([int(games[k].color=='W') for k in keys]);dest=RUN/'official';dest.mkdir()
    for pool in pools:
        ss=[[pos[k] for k in pool.supports[p]] for p in pool.candidate_ids];qs=[[pos[k] for k in q['keys']] for q in pool.questions]
        v=color_score(raw,colors,ss,qs,RUN/'final_models');pd=dest/pool.name;pd.mkdir()
        d=np.load(b.RUN/f'official_stat/{pool.name}/D.npy');seq=np.load(b.RUN/f'official_stat/{pool.name}/sequence.npy')
        e=fixed_rrf60(d,v);f=fixed_rrf60(e,seq)
        for name,s in [('color_PLDA',v),('E_C',e),('F_C',f)]:np.save(pd/f'{name}.npy',s,allow_pickle=False)
    b.write(dest/'RECEIPT.json',{'no_quality_labels':True,'calibration_protocol':b.sha(RUN/'PROTOCOL.json'),'final_model_receipt':b.sha(RUN/'final_models/FIT_RECEIPT.json')})

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','fit','cal-predict','select','refit','official-predict']);v=a.parse_args()
    with threadpool_limits(limits=4):{'prepare':prepare,'fit':fit_models,'cal-predict':cal_predict,'select':selection,'refit':lambda:fit_models(True),'official-predict':official_predict}[v.stage]()
