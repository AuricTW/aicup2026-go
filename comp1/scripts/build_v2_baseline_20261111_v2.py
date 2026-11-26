"""New-release baseline. Immutable old bundles; no upload or answer-key loader.

Commands are staged so source/data/gates precede quality. Official inference
reads only the released candidate/query CSVs and frozen model assets.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from pathlib import Path
import sys, json, csv, hashlib, time, re, argparse, multiprocessing
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from goid import inference as base, fixed_batch, row_projection
from goid_v4.csv_u0_v1 import extract, FrozenU0, encode_sets, scores_from_embeddings, branch_bundle, load_bundle
from goid_v4.p1 import compose_game_features
from goid_v4.plda_metric_v1 import PLDA, histograms, summarize
from goid_submission_v1.gaussian_csv import FrozenGaussian, summaries
from goid_submission_v1.ensemble_D_v2 import fixed_rrf60
from goid_submission_v1.sequence_F_v1 import openings
from goid_v4.score_push_features_v1 import design, BINS, sequence_export
RUN = ROOT/'runs/v2_baseline_20261111_v2'
RAW = ROOT/'data/raw/official_tests_20261111_v2/extracted/tests'
F_BUNDLE = ROOT/'artifacts/inference_bundle_F_E_sequence_20261104_v1'
U_BUNDLE = F_BUNDLE/'E/D/u0'
H_BUNDLE = ROOT/'artifacts/inference_bundle_H_margin_20261105_v1'
SAFE = ROOT/'runs/research_v4/official_aligned_v1_20261104/scope_metadata.npz'
R1 = ROOT/'runs/fisher_policy_pilot_20261105_v2'
FAMILIES = ('raw','move','canonical','group')

def read(p): return base.read_json(p)
def sha(p): return base.sha(p)
def write(p, x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def log(**x): print(json.dumps(x,ensure_ascii=False),flush=True)
def meta():
    with np.load(SAFE,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def ranks(a):
    order=np.argsort(-a,axis=1,kind='stable');r=np.empty_like(order)
    np.put_along_axis(r,order,np.broadcast_to(np.arange(1,a.shape[1]+1),a.shape),axis=1)
    return r
def runtime():
    import torch
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    assert 'RTX 3060' in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(.65)
    return 'cuda'

def prepare():
    RUN.mkdir(exist_ok=False)
    m=meta();old=read(R1/'episodes/ep0.json')
    positives=sorted({q['player'] for q in old['questions']});assert len(positives)==323
    excluded=set(read(ROOT/'runs/4090_support100_pilot_20261105_v1/EPISODE_A.json')['candidates'])
    pi=np.load(R1/'indices.npy')[np.load(R1/'prior_rows.npy')]
    excluded.update(m['owners'][pi].astype(str));assert not set(positives)&excluded
    # Exclude entire current A/B identity families, not only its selected rows.
    mask=np.isin(m['owners'].astype(str),sorted(excluded));ok=~mask & (m['hd']<=1)
    for f in FAMILIES:ok &= ~np.isin(m[f],np.unique(m[f][mask]))
    by=defaultdict(list)
    for i in np.flatnonzero(ok):by[m['owners'][i].decode()].append(int(i))
    # All members of an audited group receive the same support/query role.
    support_role=np.asarray([int(x[:8],16)%10<7 for x in m['group']],bool)
    def unique(ids):
        seen={f:set() for f in FAMILIES};out=[]
        for i in sorted(ids,key=lambda j:m['raw'][j]):
            if any(m[f][i] in seen[f] for f in FAMILIES):continue
            out.append(i)
            for f in FAMILIES:seen[f].add(m[f][i])
        return out
    banks={p:(unique([i for i in ids if support_role[i]]),unique([i for i in ids if not support_role[i]])) for p,ids in by.items()}
    assert all(p in banks and len(banks[p][0])>=10 and len(banks[p][1])>=5 for p in positives)
    filler=sorted((p for p in banks if p not in positives and len(banks[p][0])>=10),key=lambda p:hashlib.sha256(('v2-fill|'+p).encode()).hexdigest())[:277]
    assert len(filler)==277
    candidates=sorted(positives+filler);positive_cols=[candidates.index(p) for p in positives]
    episodes=[]
    for seed in (2026111101,2026111102):
        rng=np.random.default_rng(seed)
        ss=[sorted(rng.choice(banks[p][0],10,replace=False).tolist(),key=lambda j:m['raw'][j]) for p in candidates]
        sf={f:set(m[f][np.concatenate(ss)]) for f in FAMILIES}
        qs=[]
        # 100 questions per length, repeated identities permitted just as in release.
        for count in (1,2,3,5):
            for p in rng.choice(positives,100,replace=False).tolist():
                allowed=[i for i in banks[p][1] if all(m[f][i] not in sf[f] for f in FAMILIES)]
                assert len(allowed)>=5,p
                ids=sorted(rng.choice(allowed,count,replace=False).tolist(),key=lambda j:m['raw'][j])
                qs.append({'id':f'cal_{seed}_{count}_{len(qs):04}','indices':ids,'q':count,'target':candidates.index(p),'player':p})
        qix=np.asarray([i for q in qs for i in q['indices']]);assert len(qs)==400
        for f in FAMILIES:assert not sf[f]&set(m[f][qix]),f
        ep={'seed':seed,'candidates':candidates,'supports':ss,'questions':qs}
        episodes.append(ep);write(RUN/f'calibration/ep{len(episodes)-1}.json',ep)
    indices=np.asarray(sorted({i for ep in episodes for s in ep['supports'] for i in s}|{i for ep in episodes for q in ep['questions'] for i in q['indices']}),np.int64)
    np.save(RUN/'calibration/indices.npy',indices,allow_pickle=False)
    protocol={'created_utc':base.utc(),'release':'official 2026-11-11 v2','official_schema_receipt':sha(ROOT/'runs/reopening_day_20261111_v1/OFFICIAL_V2_SCHEMA_AUDIT.json'),
      'scope_metadata_SHA256':sha(SAFE),'local_GPU':'RTX3060 only','candidates':600,'support':10,'episodes':2,'queries_per_episode':400,'query_lengths':[1,2,3,5],
      'positive_identities':323,'distractor_identities':277,'unique_games':len(indices),
      'positive_identities_excluded_from_F_H_fitting_by_existing_scope_receipt':sha(R1/'SCOPE_AUDIT.json'),
      'distractors_may_have_been_seen_in_frozen_base_fitting':True,'not_all600_identity_disjoint':True,
      'historically_reused_development_players_not_independent_confirmation':True,
      'current_A_B_and_prior256_all_owners_and_all_four_families_excluded':True,
      'retired400_details_not_opened_source_already_excludes_them':True,'Public_labels_not_loaded':True,
      'candidate_recipes':['D','E','F','E_H','H','K_only_diagnostic'],
      'D':'equal rank RRF60(expanded color Gaussian,U0(.75colorRRF+.25rankMeanCount))',
      'E':'equal rank RRF60(D,raw histogram PLDA)','F':'equal rank RRF60(E,target-relative opening sequence)',
      'H':'.9*F+.1/(60+rankHumanSL)','E_H':'.9*E+.1/(60+rankHumanSL)',
      'HumanSL_white_extension':'actual target decisions, actual side-to-play metadata, +draw-adjustment for W; no board recoloring',
      'selection_base':'D','promotion_gate':{'mean_exponential_delta_gte':.003,'each_episode_delta_gte':0,'q1_delta_gte':0,'q2_delta_gte':0,'Top1_net_gte':0,'shared_player_bootstrap_95_lower_gt':0,'bootstrap_draws':5000,'seed':2026111103},
      'if_multiple_pass':'highest mean exponential score; lexical name tie-break','if_none_pass':'D',
      'old800':'reuse frozen model-generated H after byte equality of original four input files','official_total_rows':1600,
      'no_one_to_one_assignment':True,'new_test_labels_not_available':'no public2/private2 accuracy claim',
      'training':False,'upload':False,'calibration_models':'paired exclusion-trained statistics+MLP/U0/Gaussian/rawPLDA/sequence, not fulltrain release weights','inference_models':'existing fulltrain frozen release assets; recipe transfer, not exact weight validation','rejected_v1':'mixed validation feature/release weight diagnostic discarded, not a model comparison'}
    write(RUN/'PROTOCOL.json',protocol)
    write(RUN/'SOURCE_FREEZE.json',{'script':str(Path(__file__)),'sha256':sha(__file__),'H_bundle':sha(H_BUNDLE/'MANIFEST.json'),'F_bundle':sha(F_BUNDLE/'manifest.json')})
    log(stage='prepared',games=len(indices),positives=323,candidates=600)

def sequence_scores(codes,colors,ss,qs,calibration=False):
    # Relative target color changes only token side, never coordinates or stones.
    codes=codes.copy();w=np.flatnonzero(colors==1)
    for i in w:
        valid=codes[i]!=65535;codes[i,valid]=(codes[i,valid]+362)%724
    with np.load((ROOT/'runs/nearperfect_local_pilot_20261105_v1/P0_predictions/frozen_refit_background.npz') if calibration else F_BUNDLE/'background.npz',allow_pickle=False) as z:bg=z['background']
    variant=read(F_BUNDLE/'manifest.json')['variant'];x=design(codes,variant)
    from scipy import sparse
    groups=len(bg)//BINS
    bank=sparse.vstack([sparse.csr_matrix(x[sorted(s)].sum(0)) for s in ss],format='csr',dtype=np.float64)
    norm=np.stack([np.asarray(bank[:,g*BINS:(g+1)*BINS].sum(1)).ravel() for g in range(groups)],1)
    bank.data=np.log1p(bank.data/(20*bg[bank.indices]));used=sorted({i for q in qs for i in q});qx=x[used]
    valid=np.stack([np.asarray(qx[:,g*BINS:(g+1)*BINS].sum(1)).ravel() for g in range(groups)],1)
    gs=(qx@bank.T).toarray()-valid@np.log1p(norm/20).T;pos={i:j for j,i in enumerate(used)}
    return np.stack([gs[[pos[i] for i in sorted(q)]].mean(0)/groups for q in qs])

def color_scores(features,colors,ss,qs):
    # The exact frozen profile/projection implementation supports actual600.
    n=len(features);keys=[f'{i:09}' for i in range(n)];games={k:base.Game(k,k,'BW'[colors[i]],'') for i,k in enumerate(keys)}
    pool=base.Pool('pool',[str(i) for i in range(len(ss))],{str(i):[keys[j] for j in s] for i,s in enumerate(ss)},[{'keys':[keys[j] for j in q]} for q in qs],{})
    pos={k:i for i,k in enumerate(keys)}
    a,b,_=row_projection.component_scores(features[:,:1582],pool,games,pos)
    c,d,_=row_projection.component_scores(features[:,1582:1710],pool,games,pos)
    original=base.rrf(a,c);return base.choose_scores(b,d,original,'color')[0]

def stat_scores(features,raw,colors,codes,ss,qs,calibration=False):
    import torch
    saved=torch.load((ROOT/'runs/score_push_experiments_20261104_v2/models/U0_mean_count_s31415/model.pt') if calibration else U_BUNDLE/'u0/model.pt',map_location='cpu',weights_only=True)
    model=FrozenU0(saved['floor']).to('cuda');model.load_state_dict(saved['state_dict']);model.eval().requires_grad_(False)
    xx=torch.from_numpy(np.asarray(features,np.float32)).to('cuda');cc=torch.from_numpy(colors.astype(np.int64)).to('cuda')
    sm,sv=encode_sets(model,xx,cc,ss);qm,qv=encode_sets(model,xx,cc,qs)
    uc=scores_from_embeddings(qm,qv,sm,sv,saved['variance_scale'])
    color=color_scores(features,colors,ss,qs);u=.75*color+.25/(60+ranks(uc))
    ga=FrozenGaussian((ROOT/'runs/score_push_experiments_20261104_v2/metrics/D_Gaussian.npz') if calibration else F_BUNDLE/'E/D/gaussian/u1/model.npz');z=ga.transform(features)
    g=ga.score(summaries(z,colors,qs),summaries(z,colors,ss));d=fixed_rrf60(u,g)
    pl=PLDA((ROOT/'runs/score_push_experiments_20261104_v2/metrics/PLDA_raw.npz') if calibration else F_BUNDLE/'E/PLDA_raw.npz');rz=pl.transform(histograms(raw,read(F_BUNDLE/'E/feature_schema.json'),sqrt=True))
    sm,sn=summarize(rz,ss);qm,qn=summarize(rz,qs);p=pl.score(qm,qn,sm,sn)
    e=fixed_rrf60(d,p);seq=sequence_scores(codes,colors,ss,qs,calibration);f=fixed_rrf60(e,seq)
    return {'D':d,'E':e,'F':f,'color':color,'U0':u,'raw_PLDA':p,'sequence':seq,'Gaussian':g}

def cal_stat():
    runtime();m=meta();ix=np.load(RUN/'calibration/indices.npy');pos={int(i):j for j,i in enumerate(ix)}
    features=np.array(np.load(ROOT/'runs/score_push_experiments_20261104_v2/features.npy',mmap_mode='r')[ix],np.float32)
    raw=np.array(np.load(ROOT/'data/processed/features.npy',mmap_mode='r')[m['rows'][ix]],np.float32)
    codes=sequence_export(ix,m);colors=m['colors'][ix]
    for n in range(2):
        ep=read(RUN/f'calibration/ep{n}.json');ss=[[pos[i] for i in s] for s in ep['supports']];qs=[[pos[i] for i in q['indices']] for q in ep['questions']]
        scores=stat_scores(features,raw,colors,codes,ss,qs,calibration=True)
        d=RUN/f'calibration/stats{n}';d.mkdir()
        for name,value in scores.items():np.save(d/f'{name}.npy',value,allow_pickle=False)
        log(stage='cal_stat',episode=n,results={k:metrics(v,ep)['score'] for k,v in scores.items() if k in ('D','E','F')})

def metrics(score,ep):
    targets=np.asarray([q['target'] for q in ep['questions']]);r=ranks(score)[np.arange(len(targets)),targets]
    value=np.where(r<=5,np.exp(-(r-1)),0.)
    return {'score':float(value.mean()),'top1':int((r==1).sum()),'top5':int((r<=5).sum()),'n':len(r),
      'by_games':{str(q):{'score':float(value[np.asarray([x['q']==q for x in ep['questions']])].mean()),'top1':int(((r==1)&np.asarray([x['q']==q for x in ep['questions']])).sum()),'n':100} for q in (1,2,3,5)}}

def load_pool_new(spec,games,bundle):
    # New1/2/3/5 contract. Original parser's5..20 guard remains untouched.
    supports={};gids=set();questions=[];seen=set()
    for _,r in base.rows_csv(spec['candidates_csv'],base.CANDIDATE_HEADER):
        p=base.identity(r['player_id'],'candidate');gid=base.identity(r['game_id'],'game')
        assert (p,gid) not in gids;gids.add((p,gid))
        k=base.add_game(games,r['sgf_content'],r['color'],bundle)
        ids=supports.setdefault(p,[]);assert k not in ids;ids.append(k)
    assert len(supports)==600 and all(len(v)==10 for v in supports.values())
    for _,r in base.rows_csv(spec['queries_csv'],base.QUERY_HEADER):
        q=base.identity(r['question_id'],'question');assert q not in seen;seen.add(q)
        n=int(r['num_games']);assert str(n)==r['num_games'] and n in (1,2,3,5)
        keys=[base.add_game(games,r[f'sgf_{i}'],r[f'color_{i}'],bundle) for i in range(1,n+1)]
        assert len(set(keys))==n
        assert all(not r[f'sgf_{i}'] and not r[f'color_{i}'] for i in range(n+1,21))
        questions.append({'question_id':q,'keys':keys,'num_games':n})
    assert len(questions)==400
    source={k:spec[k] for k in ('candidates_csv','queries_csv')}
    source.update({k+'_sha256':sha(spec[k+'_csv']) for k in ('candidates','queries')})
    return base.Pool(spec['name'],list(supports),supports,questions,source)

def load_official():
    games={};pools=[];b={'manifest':{'parser_sha256':sha(ROOT/'src/goid/sgf_features.py'),'runtime':{'sgfmill':'1.1.1'}},'banned_raw':set(),'banned_players':set(),'banned_canonical':set()}
    for side in ('public','private'):
        spec={'name':side+'2','candidates':str(RAW/f'player_identification_candidates2_{side}.csv'),'queries':str(RAW/f'player_identification_test2_{side}.csv'),'expected_candidates':600,'reference_games':10}
        # base loader's field names are an existing parser contract.
        spec['candidates_csv']=spec.pop('candidates');spec['queries_csv']=spec.pop('queries')
        pool=load_pool_new(spec,games,b)
        pool.candidate_ids.sort();pool.questions.sort(key=lambda q:q['question_id'])
        for p in pool.candidate_ids:pool.supports[p].sort()
        for q in pool.questions:q['keys'].sort();assert len(q['keys']) in (1,2,3,5)
        pools.append(pool)
    return games,pools

def official_stat():
    runtime();games,pools=load_official();d=RUN/'official_stat';d.mkdir()
    raw,keys,met,cache=extract(games,RUN/'cache/stats.sqlite',d,4)
    bundle=load_bundle(U_BUNDLE);(stat,mlp),evidence=fixed_batch.encode_features(raw,branch_bundle(bundle,'refit'),'cuda')
    count=np.asarray([met[k]['target_moves'] for k in keys],np.float64)
    features=compose_game_features(stat,mlp,raw[:,:12],[games[k].color for k in keys],[met[k]['mainline_moves'] for k in keys],np.rint(raw[:,9:12].astype(np.float64)*count[:,None]))
    codes,other,seq_report=openings(games,RUN/'cache/openings',4);assert other==keys
    colors=np.asarray([int(games[k].color=='W') for k in keys],np.int64);pos={k:i for i,k in enumerate(keys)}
    for pool in pools:
        ss=[[pos[k] for k in pool.supports[p]] for p in pool.candidate_ids];qs=[[pos[k] for k in q['keys']] for q in pool.questions]
        scores=stat_scores(features,raw,colors,codes,ss,qs);pd=d/pool.name;pd.mkdir()
        for name,v in scores.items():np.save(pd/f'{name}.npy',v,allow_pickle=False)
        write(pd/'mapping.json',{'candidates':pool.candidate_ids,'questions':[q['question_id'] for q in pool.questions],'supports':ss,'queries':qs})
    np.save(d/'features.npy',features,allow_pickle=False);write(d/'keys.json',keys)
    write(d/'RECEIPT.json',{'games':len(keys),'cache':cache,'encoder':evidence,'opening':seq_report,'no_labels':True})
    log(stage='official_stat_complete',games=len(keys))

def encode_target(item):
    # Dedicated mixed-color adapter; frozen original encode_game stays unchanged.
    index,text,expected,hd,target,config=item
    from goid_kg_v1 import native as n
    from katago.game.board import Board
    from katago.game.features import Features
    from katago.game.gamestate import GameState
    from goid.sgf_features import analyze_sgf,decode_sequence
    from goid_v4.replay import select_plies
    assert hashlib.sha256(text.encode()).hexdigest()==expected and target in ('B','W') and hd in (0,1)
    parsed=analyze_sgf(text,target,include_features=False);moves=decode_sequence(parsed['move_sequence'],19)
    assert not decode_sequence(parsed['setup_sequence'],19)
    chosen=select_plies(moves,target.lower(),parsed['canonical_hash'],quotas=(4,4,0),seed=20261104)
    if not chosen:return index,None,'no_target_decision'
    rules=GameState.RULES_JAPANESE.copy();rules['whiteKomi']=6.5 if hd==0 else 0.
    gs=GameState(19,rules);fe=Features(config,19);metadata=n.profile_metadata('rank_1d').get_metadata_row(target,361)
    packed=[];glob=[];legal=[];actions=[];scalars=[];previous_pass=False;double_pass=False
    try:
        for ply,(color,point) in enumerate(moves,1):
            pla=Board.BLACK if color=='b' else Board.WHITE
            loc=Board.PASS_LOC if point is None else gs.board.loc(point[1],18-point[0])
            if ply in chosen:
                if double_pass:return index,None,'encore_after_double_pass'
                assert color==target.lower() and gs.board.pla==pla
                gs.rules['passWouldEndPhase']=previous_pass;x,g=gs.get_input_features(fe)
                if hd==1:g[0,5]+=(.5 if target=='W' else -.5)/20.
                assert ((x==0)|(x==1)).all() and np.isfinite(g).all()
                mask=np.asarray([gs.board.would_be_legal(pla,fe.tensor_pos_to_loc(i,gs.board)) and not gs.board.would_be_suicide(pla,fe.tensor_pos_to_loc(i,gs.board)) for i in range(361)]+[True])
                action=361 if point is None else (18-point[0])*19+point[1];assert mask[action]
                packed.append(np.packbits(x[0].reshape(-1).astype(np.uint8)));glob.append(g[0]);legal.append(np.packbits(mask));actions.append(action)
                scalars.append([float(ply<=60),float(ply>60),min(ply/180,1),float(hd),float(action==361)])
            if ply>=max(chosen):break
            gs.play(pla,loc);double_pass=double_pass or (previous_pass and point is None);previous_pass=point is None
    except Exception as e:return index,None,f'{type(e).__name__}:{e}'
    return index,(np.stack(packed),np.stack(glob),np.stack(legal),np.asarray(actions,np.int64),np.asarray(scalars,np.float32),metadata),None

def encode_batch(items):return [encode_target(i) for i in items]

def human_tasks(kind):
    from goid_kg_v1 import native as n
    config=read(n.RUN/'model_config.json')
    if kind=='calibration':
        from goid_v4.pilot_scope_v1 import PointReader
        m=meta();ix=np.load(RUN/'calibration/indices.npy');reader=PointReader(ROOT/'data/processed/audit.sqlite')
        try:
            for start in range(0,len(ix),32):
                ids=ix[start:start+32];texts={r['row_id']:r['sgf_content'] for r in reader.sample_text(m['rows'][ids])}
                yield [(start+j,texts[int(m['rows'][i])],m['raw'][i].decode(),int(m['hd'][i]),'BW'[m['colors'][i]],config) for j,i in enumerate(ids)]
        finally:reader.close()
    else:
        games,_=load_official();keys=read(RUN/'official_stat/keys.json')
        for start in range(0,len(keys),32):
            batch=[]
            for j,k in enumerate(keys[start:start+32]):
                g=games[k];hd=int(re.match(r'^\(;SZ\[19\]HD\[([01])\];',g.sgf).group(1));batch.append((start+j,g.sgf,g.raw_hash,hd,g.color,config))
            yield batch

def human_extract(kind):
    import torch
    from goid_kg_v1 import native as n
    from goid_kg_v1.features import infer_games,encode_game
    runtime();d=RUN/(kind+'_human');d.mkdir()
    count=len(np.load(RUN/'calibration/indices.npy')) if kind=='calibration' else len(read(RUN/'official_stat/keys.json'))
    arr=np.lib.format.open_memmap(d/'vectors.npy',mode='w+',dtype=np.float32,shape=(count,2339));done=np.zeros(count,bool)
    start=time.perf_counter();model=n.load_backbone('cuda',False)
    # Engineering check: the B extension must exactly reproduce the original encoder.
    batches=human_tasks(kind);first=next(batches);checks=[]
    for item in first:
        if item[4]=='B':
            a=encode_target(item);b=encode_game(item[:4]+(item[5],));assert a[2]==b[2]
            assert a[1] is not None
            assert all(np.array_equal(x,y) for x,y in zip(a[1],b[1]));checks.append({'index':item[0],'B_exact':True})
            if len(checks)>=2:break
    assert checks
    # Fixed32: each game is evaluated on its own, padded by that game's first
    # position, so other games/query selection/order cannot change GPU neighbors.
    def single(record):return infer_games(model,[record],batch_size=32)[record[0]]
    probes=[encode_target(x) for x in first[:4]]
    assert all(x[1] is not None and x[2] is None for x in probes)
    pa={x[0]:single(x) for x in probes};pb={x[0]:single(x) for x in reversed(probes)}
    assert all(np.array_equal(pa[k],pb[k]) for k in pa)
    write(d/'ENGINEERING.json',{'B_exact_original_encoder':checks,'probe_cold_warm_subset_order_exact':True,'fixed_GPU_batch':32,'one_game_per_call':True,'no_cross_game_neighbors':True,'white_HD1_adjustment':'+0.5/20','white_uses_actual_player_metadata':True})
    import itertools
    iterator=iter(itertools.chain([first],batches))
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending=set()
        for _ in range(8):
            batch=next(iterator,None)
            if batch:pending.add(pool.submit(encode_batch,batch))
        while pending:
            ready,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                records=future.result()
                for record in records:
                    assert record[1] is not None and record[2] is None,(record[0],record[2])
                    assert not done[record[0]];arr[record[0]]=single(record);done[record[0]]=True
                if int(done.sum())%256==0 or done.all():
                    arr.flush();log(stage=kind+'_HumanSL',games=int(done.sum()),total=count,seconds=time.perf_counter()-start)
                batch=next(iterator,None)
                if batch:pending.add(pool.submit(encode_batch,batch))
    assert done.all();arr.flush()
    write(d/'RECEIPT.json',{'games':count,'all_valid':True,'seconds':time.perf_counter()-start,'vectors_SHA256':sha(d/'vectors.npy'),'training':False,'profile':'rank_1d','SWA':False,'precision':'FP16','old_Black_fitted_PLDA_white_transfer_requires_calibration_gate':True})

def human_score(kind):
    d=RUN/(kind+'_human');raw=np.load(d/'vectors.npy',mmap_mode='r')
    with np.load(H_BUNDLE/'preprocess.npz',allow_pickle=False) as z:x=np.clip((np.asarray(raw,np.float64)-z['mean'])/z['std'],-10,10).astype(np.float32)
    model=PLDA(H_BUNDLE/'trunk_PLDA.npz');z=model.transform(x)
    for n in range(2):
        if kind=='calibration':
            ix=np.load(RUN/'calibration/indices.npy');pos={int(i):j for j,i in enumerate(ix)};ep=read(RUN/f'calibration/ep{n}.json')
            ss=[[pos[i] for i in s] for s in ep['supports']];qs=[[pos[i] for i in q['indices']] for q in ep['questions']];source=RUN/f'calibration/stats{n}';dest=d/f'ep{n}'
        else:
            name=('public2','private2')[n];mapping=read(RUN/f'official_stat/{name}/mapping.json');ss=mapping['supports'];qs=mapping['queries'];source=RUN/f'official_stat/{name}';dest=d/name
        dest.mkdir();sm,sn=summarize(z,ss);qm,qn=summarize(z,qs);k=model.score(qm,qn,sm,sn)
        np.save(dest/'K.npy',k,allow_pickle=False)
        for label,prior in (('H','F'),('E_H','E')):
            value=.9*np.load(source/f'{prior}.npy')+.1/(60+ranks(k));np.save(dest/f'{label}.npy',value,allow_pickle=False)
        # Query order/subset prediction stability, not a fresh quality estimate.
        take=np.arange(0,len(qs),37);sub=model.score(qm[take[::-1]],qn[take[::-1]],sm,sn)
        assert np.array_equal(k[take[::-1]],sub)
    write(d/'SCORING_COMPLETE.json',{'fixed_PLDA':sha(H_BUNDLE/'trunk_PLDA.npz'),'query_subset_order_exact':True,'score_training':False})

def select():
    names=('D','E','F','E_H','H');allm={};values={};episodes=[read(RUN/f'calibration/ep{n}.json') for n in range(2)]
    for name in names:
        vv=[];mm=[]
        for n,ep in enumerate(episodes):
            p=RUN/(f'calibration/stats{n}' if name in ('D','E','F') else f'calibration_human/ep{n}')/f'{name}.npy'
            s=np.load(p);t=np.asarray([q['target'] for q in ep['questions']]);r=ranks(s)[np.arange(len(t)),t]
            vv.append((np.where(r<=5,np.exp(-(r-1)),0),r));mm.append(metrics(s,ep))
        values[name]=vv;allm[name]=mm
    players=sorted({q['player'] for ep in episodes for q in ep['questions']});playerpos={p:i for i,p in enumerate(players)}
    rng=np.random.default_rng(2026111103);draw=rng.integers(len(players),size=(5000,len(players)));gates={};passing=[]
    for name in names[1:]:
        differences=[values[name][n][0]-values['D'][n][0] for n in range(2)];diff=np.concatenate(differences)
        sums=np.zeros(len(players));counts=np.zeros(len(players));net=0
        for n,ep in enumerate(episodes):
            for q,v in zip(ep['questions'],differences[n]):i=playerpos[q['player']];sums[i]+=v;counts[i]+=1
            net+=int((values[name][n][1]==1).sum()-(values['D'][n][1]==1).sum())
        bootstrap=sums[draw].sum(1)/counts[draw].sum(1);ci=np.quantile(bootstrap,[.025,.975]).tolist()
        byq={q:float(np.concatenate([v[np.asarray([x['q']==q for x in ep['questions']])] for v,ep in zip(differences,episodes)]).mean()) for q in (1,2,3,5)}
        passed=diff.mean()>=.003 and all(v.mean()>=0 for v in differences) and byq[1]>=0 and byq[2]>=0 and net>=0 and ci[0]>0
        gates[name]={'delta':float(diff.mean()),'episode_delta':[float(v.mean()) for v in differences],'by_q_delta':byq,'top1_net':net,'player_cluster95CI':ci,'passed':bool(passed)}
        if passed:passing.append(name)
    selected=sorted(passing,key=lambda n:(-np.mean([v['score'] for v in allm[n]]),n))[0] if passing else 'D'
    write(RUN/'SELECTION.json',{'selected':selected,'metrics':allm,'gates_vs_D':gates,'scope':'two reused-player development episodes,600 candidates x10support,400 queries each; not independent, official or Private','Public2_labels_loaded':False})
    log(stage='selected',selected=selected,mean_scores={k:float(np.mean([v['score'] for v in mm])) for k,mm in allm.items()},gates=gates)

def finalize():
    chosen=read(RUN/'SELECTION.json')['selected'];games,pools=load_official()
    old=ROOT/'submissions/20261105_H_margin_HumanSL_v1/07_H_F_HumanSL_margin.csv'
    assert sha(old)=='0a6c9bf7e7af03e9623818be696cbca4bb4426776fdbabbbdaa5eb61523a69a7'
    receipt=read(ROOT/'runs/reopening_day_20261111_v1/OFFICIAL_V2_SCHEMA_AUDIT.json')
    # Independently re-verify immutable original four inputs before reuse.
    oldraw=ROOT/'data/raw/official_tests_20261104_v1/tests'
    if not oldraw.exists():
        matches=list((ROOT/'data/raw').rglob('player_identification_test_public.csv'))
        oldraw=next(p.parent for p in matches if '20261104' in str(p))
    for part in ('candidates','test'):
        for side in ('public','private'):
            file=f'player_identification_{part}_{side}.csv';assert sha(RAW/file)==sha(oldraw/file),(file,oldraw)
    predictions={r['question_id']:[r[f'top_{i}'] for i in range(1,6)] for _,r in base.rows_csv(old,base.HEADER)}
    assert len(predictions)==800
    for pool in pools:
        source=RUN/(f'official_stat/{pool.name}' if chosen in ('D','E','F') else f'official_human/{pool.name}')/f'{chosen}.npy'
        scores=np.load(source);assert scores.shape==(400,600) and np.isfinite(scores).all()
        order=np.argsort(-scores,axis=1,kind='stable')[:,:5]
        for q,cols in zip(pool.questions,order):
            assert q['question_id'] not in predictions
            predictions[q['question_id']]=[pool.candidate_ids[i] for i in cols]
    qids=[r['question_id'] for _,r in base.rows_csv(RAW/'sample_submission_player_v2.csv',base.HEADER)]
    assert len(qids)==1600 and len(set(qids))==1600 and set(qids)==set(predictions)
    dest=ROOT/'submissions/20261111_v2_baseline_v1';dest.mkdir(exist_ok=False)
    csvfile=dest/f'submission_player_v2_H_old800_{chosen}_new800.csv'
    with csvfile.open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(base.HEADER)
        for q in qids:writer.writerow([q]+predictions[q])
    check=validate(csvfile)
    assert check['valid']
    out={'CSV':str(csvfile),'bytes':csvfile.stat().st_size,'SHA256':sha(csvfile),'rows':1600,'old800':'frozen H, unchanged model-generated predictions','new800':chosen,'format':check,'official_score':None,'Public2_accuracy':None,'uploaded':False,'no_training_on_test_or_answer_lookup':True,'selection':str(RUN/'SELECTION.json')}
    write(dest/'RECEIPT.json',out);write(RUN/'DELIVERY.json',out)
    write(RUN/'MANIFEST_SHA256.json',{p.relative_to(RUN).as_posix():sha(p) for p in RUN.rglob('*') if p.is_file() and 'cache' not in p.relative_to(RUN).parts})
    log(stage='ready',**out)

def validate(path):
    qtopool={};members={}
    for number in ('','2'):
        for side in ('public','private'):
            pool=side+number
            members[pool]={r['player_id'] for _,r in base.rows_csv(RAW/f'player_identification_candidates{number}_{side}.csv',base.CANDIDATE_HEADER)}
            for _,r in base.rows_csv(RAW/f'player_identification_test{number}_{side}.csv',base.QUERY_HEADER):
                assert r['question_id'] not in qtopool;qtopool[r['question_id']]=pool
    seen=set();counts=Counter()
    for _,r in base.rows_csv(path,base.HEADER):
        assert list(r)==base.HEADER
        q=r['question_id'];assert q in qtopool and q not in seen;seen.add(q)
        pp=[r[f'top_{i}'] for i in range(1,6)];assert len(set(pp))==5 and all(p in members[qtopool[q]] for p in pp)
        counts[qtopool[q]]+=1
    assert seen==set(qtopool) and len(seen)==1600 and all(v==400 for v in counts.values())
    return {'valid':True,'rows':len(seen),'by_pool':dict(counts),'header':base.HEADER,'UTF8':True,'exact_question_set':True,'all5_unique_in_correct_actual_candidate_pool':True,'sample_row_order':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','cal-stat','official-stat','cal-human','official-human','cal-score','official-score','select','finalize']);a=ap.parse_args()
    with threadpool_limits(limits=4):
        {'prepare':prepare,'cal-stat':cal_stat,'official-stat':official_stat,'cal-human':lambda:human_extract('calibration'),'official-human':lambda:human_extract('official'),'cal-score':lambda:human_score('calibration'),'official-score':lambda:human_score('official'),'select':select,'finalize':finalize}[a.stage]()
if __name__=='__main__':main()
