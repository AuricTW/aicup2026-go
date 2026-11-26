"""Versioned point-lookup export for the explicitly authorized real P1 pilot.

Never calls the old scope builders, reads retired lists, scans games, or hashes
the mixed DB/feature payload. Only the frozen fit row list and declared dev rows
can be returned by SQLite or indexed from the feature mmap.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,sqlite3,time
import numpy as np
from .artifacts import ContractError,object_hash

RANKS=('D','C','B','A','1D','2D','3D','4D','5D','6D')
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/research_v4/p1_pilot_v1_20261026'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def fit_player(player):
    digest=hashlib.blake2b(f'20261025\0player:{player}'.encode(),digest_size=16).digest()
    return int.from_bytes(digest,'big')/2**128 < .6


def opaque(player):return hashlib.sha256(('goid-v4-player-v1|'+player).encode()).hexdigest()


class PointReader:
    columns=('row_id,source_file,source_row,game_id,player_id,rank,color,parse_ok,'
             'raw_hash,move_hash,canonical_hash,COALESCE(split_group_hash,canonical_hash) AS group_key,'
             'mainline_moves,target_moves')
    def __init__(self,path):
        self.path=Path(path)
        self.db=sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA query_only=ON')
        self.queries=0;self.returned=0;self.plan=None

    def fetch(self,row_ids):
        ids=[int(x) for x in row_ids]
        if len(ids)!=len(set(ids)) or any(i<0 for i in ids):raise ContractError('Invalid explicit row allowlist')
        result=[]
        for start in range(0,len(ids),800):
            batch=ids[start:start+800]
            sql=f'SELECT {self.columns} FROM games WHERE row_id IN ({",".join("?" for _ in batch)})'
            plan=[r[3] for r in self.db.execute('EXPLAIN QUERY PLAN '+sql,batch)]
            if not plan or any('SCAN' in x.upper() for x in plan) or not any('SEARCH games USING INTEGER PRIMARY KEY' in x for x in plan):
                raise ContractError('Source query is not a primary-key point lookup')
            self.plan=plan
            rows=[dict(r) for r in self.db.execute(sql,batch)]
            if {r['row_id'] for r in rows}!=set(batch):raise ContractError('Missing or extra source row')
            if any(r['parse_ok']!=1 for r in rows):raise ContractError('Unparsed source row')
            result.extend(rows);self.queries+=1;self.returned+=len(rows)
        return result

    def close(self):self.db.close()

    def sample_text(self,ids):
        batch=[int(x) for x in ids]
        if len(batch)>800 or len(set(batch))!=len(batch):raise ContractError('Invalid verification sample')
        sql=f'SELECT row_id,sgf_content,color FROM games WHERE row_id IN ({",".join("?" for _ in batch)})'
        plan=[r[3] for r in self.db.execute('EXPLAIN QUERY PLAN '+sql,batch)]
        if any('SCAN' in x.upper() for x in plan):raise ContractError('Sample query would scan')
        rows=[dict(r) for r in self.db.execute(sql,batch)]
        if {r['row_id'] for r in rows}!=set(batch):raise ContractError('Verification sample incomplete')
        self.queries+=1;self.returned+=len(rows)
        return rows


def dev_plans():
    pools=[];owners={}
    allowed=set(json.loads((ROOT/'runs/development_protocol_v1/allowed_players.json').read_text(encoding='utf-8')))
    for seed in range(20261027,20261032):
        path=ROOT/f'runs/development_protocol_v1/development_s{seed}/episodes.json'
        ep=json.loads(path.read_text(encoding='utf-8'))
        if len(ep['candidate_ids'])!=400 or not set(ep['candidate_ids'])<=allowed:raise ContractError('Dev candidate scope changed')
        for p,ids in ep['support_row_ids'].items():
            for row in ids:
                if row in owners and owners[row]!=p:raise ContractError('Dev owner conflict')
                owners[row]=p
        for q in ep['questions']:
            for row in q['query_row_ids']:
                if row in owners and owners[row]!=q['player_id']:raise ContractError('Query owner conflict')
                owners[row]=q['player_id']
        pools.append({'name':f'development_s{seed}','candidate_ids':ep['candidate_ids'],
                      'support_row_ids':ep['support_row_ids'],'questions':ep['questions'],
                      'source_sha256':sha(path)})
    return pools,owners


def select_fit(rows,config):
    by_player=defaultdict(list)
    groups=set()
    for row in rows:
        if not fit_player(row['player_id']) or row['group_key'] in groups:raise ContractError('Frozen fit ownership/group invalid')
        groups.add(row['group_key']);by_player[row['player_id']].append(row)
    by_rank=defaultdict(list)
    for p,rs in by_player.items():
        ranks={r['rank'] for r in rs}
        if len(rs)>=120 and len(ranks)==1:by_rank[next(iter(ranks))].append(p)
    chosen=[];counts={}
    for rank in RANKS:
        ps=sorted(by_rank[rank],key=lambda p:object_hash({'seed':config['selection_seed'],'player':p}))
        ntrain=config['fit_train_players_per_rank'];ncal=config['fit_cal_players_per_rank']
        if len(ps)<ntrain+ncal:raise ContractError(f'Insufficient fit-only capacity: {rank}')
        counts[rank]=len(ps)
        for i,p in enumerate(ps[:ntrain+ncal]):
            part='train' if i<ntrain else 'cal'
            games=sorted(by_player[p],key=lambda r:object_hash({'seed':config['selection_seed'],'group':r['group_key']}))[:120]
            chosen.extend(dict(g,part=part) for g in games)
    return chosen,counts


def export():
    start=time.perf_counter();out=RUN/'data';out.mkdir(exist_ok=False)
    config=json.loads((ROOT/'configs/research_v4/p1_pilot_v1_20261026.json').read_text(encoding='utf-8'))
    ids_path=ROOT/'runs/statistical_v1/fit_row_ids.npy'
    if sha(ids_path)!=config['fit_rows_sha256']:raise ContractError('Frozen fit index changed')
    ids=np.load(ids_path,allow_pickle=False)
    if ids.shape!=(588294,) or ids.dtype!=np.int64 or len(np.unique(ids))!=len(ids):raise ContractError('Frozen fit shape/dtype invalid')
    pools,owners=dev_plans()
    if set(ids.tolist())&set(owners):raise ContractError('Fit/dev row intersection')
    db=ROOT/'data/processed/audit.sqlite'
    wal=db.with_name(db.name+'-wal')
    if wal.exists() and wal.stat().st_size:raise ContractError('Live source WAL requires separate immutable snapshot, not checkpointing old DB')
    before=(db.stat().st_size,db.stat().st_mtime_ns)
    reader=PointReader(db)
    try:
        fitrows=reader.fetch(ids)
        selected,counts=select_fit(fitrows,config);del fitrows
        dev=reader.fetch(sorted(owners))
        if any(r['player_id']!=owners[r['row_id']] or fit_player(r['player_id']) for r in dev):raise ContractError('Dev row/player ownership conflict')
        fitplayers={r['player_id'] for r in selected};devplayers={r['player_id'] for r in dev}
        if fitplayers&devplayers:raise ContractError('Fit/dev players overlap')
        for key in ('raw_hash','move_hash','canonical_hash','group_key'):
            if {r[key] for r in selected}&{r[key] for r in dev}:raise ContractError(f'Known fit/dev {key} overlap')
        selected.extend(dict(r,part='dev') for r in dev)
        manifest=json.loads((ROOT/'artifacts/inference_bundle_v1/manifest.json').read_text(encoding='utf-8'))
        for r in selected:
            r['key']=hashlib.sha256(f"goid-feature-cache-v1|{manifest['parser_sha256']}|1.1.1|{r['raw_hash']}|{r['color']}".encode()).hexdigest()
        selected.sort(key=lambda r:r['key'])
        if len({r['key'] for r in selected})!=len(selected):raise ContractError('Repeated exported content key')
        if (db.stat().st_size,db.stat().st_mtime_ns)!=before:raise ContractError('Source changed during point lookup')
        report={'fit_index_sha256':sha(ids_path),'fit_row_index_count':len(ids),'fit_capacity_by_rank':counts,
                'selected_counts':dict(Counter(r['part'] for r in selected)),'fit_selected_players':len(fitplayers),
                'dev_players':len(devplayers),'SQL_queries':reader.queries,'SQL_returned_rows':reader.returned,
                'query_plan':reader.plan,'full_table_scan':False,'source_DB_full_hash_recomputed':False,
                'source_feature_full_hash_recomputed':False,'retired_files_read':False,
                'retired_exclusion_basis':'historical immutable fit-index and development protocol exclusion lineage; no fresh retired-member scan',
                'source_DB_size_mtime_unchanged':True,'G2_independence_claim':False}
    finally:reader.close()
    # Metadata and original vectors are separated into train/cal/dev artifacts.
    features=np.load(ROOT/'data/processed/features.npy',mmap_mode='r',allow_pickle=False)
    if features.shape!=(1000000,1594) or features.dtype!=np.float32:raise ContractError('Unexpected mixed feature header')
    original=np.lib.format.open_memmap(out/'selected_raw_features.npy',mode='w+',dtype=np.float32,shape=(len(selected),1594))
    for k in range(0,len(selected),2048):
        rows=[r['row_id'] for r in selected[k:k+2048]]
        block=np.array(features[rows],dtype=np.float32,copy=True)
        if not np.isfinite(block).all():raise ContractError('Nonfinite allowed feature values')
        original[k:k+len(rows)]=block
    original.flush();del original,features
    # A bounded replay spot-check verifies current selected row/features alignment.
    from goid.sgf_features import analyze_sgf
    sample=sorted(range(len(selected)),key=lambda i:object_hash({'sample':selected[i]['key']}))[:200]
    verify_reader=PointReader(db)
    raw=np.load(out/'selected_raw_features.npy',mmap_mode='r',allow_pickle=False)
    positions={selected[i]['row_id']:i for i in sample}
    try:
        for row in verify_reader.sample_text(positions):
            v=analyze_sgf(row['sgf_content'],row['color'])
            i=positions[row['row_id']]
            if not np.array_equal(v['features'],raw[i]) or any(v[k]!=selected[i][k] for k in ('raw_hash','move_hash','canonical_hash')):
                raise ContractError('Allowed-row replay/feature spot-check failed')
    finally:verify_reader.close()
    del raw
    report['selected_row_replay_spotchecks']=len(sample)
    row_to_pos={r['row_id']:i for i,r in enumerate(selected)}
    plans=[]
    for p in pools:
        plans.append({'name':p['name'],'candidate_order':[opaque(x) for x in p['candidate_ids']],
                      'supports':[[row_to_pos[row] for row in p['support_row_ids'][x]] for x in p['candidate_ids']],
                      'questions':[{'question_id':q['question_id'],'player':opaque(q['player_id']),'q':q['query_games'],
                                    'indices':[row_to_pos[row] for row in q['query_row_ids']]} for q in p['questions']],
                      'source_sha256':p['source_sha256']})
    records=[]
    for r in selected:
        records.append({'key':r['key'],'row_id':r['row_id'],'source_row_key':f"{r['source_file']}:{r['source_row']}:{r['game_id']}:{r['color']}",
                        'player':opaque(r['player_id']),'color':0 if r['color']=='B' else 1,
                        'rank':r['rank'] if r['part']!='dev' else None,'part':r['part'],
                        'raw_token':r['raw_hash'],'move_token':r['move_hash'],'canonical_token':r['canonical_hash'],
                        'effective_group_token':hashlib.sha256(r['group_key'].encode()).hexdigest(),
                        'mainline_moves':r['mainline_moves'],'target_moves':r['target_moves']})
    write(out/'allowed_records.json',records);write(out/'development_plans.json',plans)
    report.update(selected_raw_features_sha256=sha(out/'selected_raw_features.npy'),
                  allowed_records_sha256=sha(out/'allowed_records.json'),development_plans_sha256=sha(out/'development_plans.json'),
                  selected_scope_hash=object_hash([(r['row_id'],r['key'],r['part']) for r in records]),elapsed_seconds=time.perf_counter()-start)
    write(out/'scope_receipt.json',report)
    print(json.dumps({'stage':'export_complete',**report},ensure_ascii=False),flush=True)


if __name__=='__main__':export()
