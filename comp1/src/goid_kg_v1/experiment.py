"""Predeclared bounded HumanSL identity pilot. Original-fit rows only."""
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,timezone,timedelta
import time,json,hashlib
import numpy as np
from . import native as n
from goid_v4 import score_push_v1 as c

def prepare():
    assert n.read(n.RUN/'engineering_v2/ENGINEERING_PARITY.json')['passed']
    m=c.metadata();old=n.read(c.RUN/'EPISODES.json');good=(m['colors']==0)&(m['hd']<=1)
    members=defaultdict(list)
    for i in np.flatnonzero(good):members[m['owners'][i].decode()].append(int(i))
    allfit=set(m['owners'].astype(str));old723={p for p,ii in members.items() if len(ii)>=120}
    short70={p for p,ii in members.items() if 110<=len(ii)<120};assert len(short70)==70
    exposures=set(old723);sources=[]
    for rel in ('runs/research_v4/p1_pilot_v1_20261026/data/cal_records.json',
                'runs/research_v4/p1_u0_outer_v1_20261026/data/cal_records.json'):
        records=n.read(n.ROOT/rel);ps={r['player'] for r in records};assert ps<=allfit
        exposures|=ps;sources.append({'path':rel,'players':len(ps),'sha256':n.sha(n.ROOT/rel)})
    outer=n.read(n.ROOT/'runs/research_v4/p1_u0_outer_v1_20261026/data/outer_plans.json')
    for p in outer:
        ps=set(p['candidate_order']);assert ps<=allfit;exposures|=ps
    sources.append({'path':'runs/research_v4/p1_u0_outer_v1_20261026/data/outer_plans.json','players':800})
    for ep in (old,n.read(c.OLD/'EPISODES.json')):
        exposures|={r['player'] for r in ep['heads'] if r['part']=='cal'}
    exposures|=set(n.read(c.RUN/'H/TRAIN_PLAN.json')['players'])
    remaining=short70-exposures
    # Reserve only known-unopened outcomes, but exclude all70 from new training.
    excluded=old723|short70;fit=np.asarray([i for i in np.flatnonzero(good) if m['owners'][i].decode() not in excluded],np.int64)
    np.save(n.RUN/'fitB_indices.npy',fit,allow_pickle=False)
    n.write(n.RUN/'EXPOSURE_CAPACITY.json',{'source':'original588294 safe fit export','previously_opened_long_eligible':723,
        'short_eligible_110_to119_upper_bound':70,'known_quality_exposed_among70':len(short70&exposures),
        'known_not_quality_exposed_short_targets':len(remaining),'current_new_fit_excluded_players':len(excluded),
        'remaining_scope_not_pristine_due_historical_training_and_external_pretraining_unknown':True,
        'future_target_players':sorted(remaining),'exposure_sources':sources,'fitB_rows':len(fit),
        'no_retired400_details_read':True})
    # Fixed100-candidate screening subset; F400 projected ranking is only an
    # exploratory comparator, not a new independent or official-aligned result.
    subset=[]
    for rank in ('D','C','B','A','1D','2D','3D','4D','5D','6D'):
        cols=[j for j,p in enumerate(old['candidates']) if m['ranks'][old['supports'][j][0]]==rank]
        subset+=sorted(cols,key=lambda j:c.token('KGpilot100',old['candidates'][j]))[:10]
    subset=sorted(subset);selected={old['candidates'][j] for j in subset};colmap={j:k for k,j in enumerate(subset)}
    questions=[dict(q,target=colmap[q['target']]) for q in old['questions'] if q['player'] in selected]
    ep={'candidates':[old['candidates'][j] for j in subset],'supports':[old['supports'][j] for j in subset],'questions':questions}
    n.write(n.RUN/'PILOT_EPISODES.json',ep)
    rows=[j for j,q in enumerate(old['questions']) if q['player'] in selected]
    ref=np.load(c.RUN/'selection_results/E_F_joint_order_D4/scores.npy')
    np.save(n.RUN/'F_projected100_reference.npy',ref[rows][:,subset],allow_pickle=False)
    # 256 training players,24 games each. Rank-stratified metadata-only quotas.
    grouped=defaultdict(list)
    for i in fit:grouped[m['owners'][i].decode()].append(int(i))
    byrank=defaultdict(list)
    for p,ii in grouped.items():
        if len(ii)>=24:byrank[str(m['ranks'][ii[0]])].append(p)
    trainplayers=[]
    while len(trainplayers)<256:
        for rank in sorted(byrank):
            available=sorted(set(byrank[rank])-set(trainplayers),key=lambda p:c.token('KGfit256',p))
            if available:trainplayers.append(available[0])
            if len(trainplayers)==256:break
    train=[];labels=[]
    for j,p in enumerate(trainplayers):
        ii=sorted(grouped[p],key=lambda i:c.token('KGfitgame',m['raw'][i].decode()))[:24]
        train+=ii;labels.extend([j]*len(ii))
    np.save(n.RUN/'train_indices.npy',np.asarray(train,np.int64),allow_pickle=False)
    np.save(n.RUN/'train_labels.npy',np.asarray(labels,np.int64),allow_pickle=False)
    used=sorted(set(train)|{i for s in ep['supports'] for i in s}|{i for q in ep['questions'] for i in q['indices']})
    assert not set(train)&{i for s in ep['supports'] for i in s}
    np.save(n.RUN/'feature_indices.npy',np.asarray(used,np.int64),allow_pickle=False)
    n.write(n.RUN/'PROTOCOL.json',{'created_at_Taipei':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'stage':'bounded100candidate100support8decision screening; full400 confirmation only if admitted',
        'backbone':'official HumanSL base weights not SWA; fixedrank_1d; frozen,source_pin_and_LICENSE',
        'source_equivalence_selection':'base matches official .bin; SWA parity failure preserved; no identity quality used for this choice',
        'positions_per_game':8,'quotas':[4,4,0],'all_semantic_inputs':'official22spatial19global192metadata;JapaneseHD0=6.5,HD1=0',
        'feature':'mean local/global/policy-action residual;residual std;early/mid residual and policy moments;mask/counts',
        'train_players':256,'games_per_player':24,'all_new_fits_exclude_players':793,
        'variants':['scalar_PLDA','trunk_PLDA','trunk_head'],'fusion':'equal RRF60(F_projected100,eachK),no weight grid',
        'head':{'steps':1000,'hidden':512,'embedding':128,'P':16,'K':4,'seed':31415,'lr':.0003,'dropout':.2,'weight_decay':.01,'loss':'SupCon.1','checkpoint':'last'},
        'pilot_admission':{'short_delta_min':.003,'q5_delta_min':0,'q10_delta_min':-.005,'q15_q20_delta_floor':-.015},
        'pilot_not_independent_or_full400_evidence':True,
        'if_admitted':'extend to400 candidates and fresh-fit F control; freeze one winner before known-unopened short cohort; no reopen323 as new',
        'full_gate':{'short_delta_min':.005,'q5_q10_delta_each_min':0,'q15_q20_delta_floor':-.005,'player_cluster95_lower_gt':0},
        'if_no_variant_admitted':'preserve negative results; no new HumanSL CSV; keep F',
        'external_pretraining_overlap_unknown_permitted_Q4':True,'no_test_adaptation_or_assignment':True,
        'no_retired400_access_or_upload':True,'feature_indices_sha256':n.sha(n.RUN/'feature_indices.npy'),
        'source_code_sha256':n.sha(Path(__file__)),'numerical_policy':'FP16 frozen backbone fixed32 rows duplicate-first tail padding;FP32 learned head; CPU summaries FP64 perquery'})
    n.log(stage='metadata_protocol_ready',feature_games=len(used),train_games=len(train),fresh_short_targets=len(remaining))
