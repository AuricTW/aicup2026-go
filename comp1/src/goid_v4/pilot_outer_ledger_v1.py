"""Post-run exposure/episode closure, never a backdated preflight receipt."""
import hashlib,json
from .pilot_outer_v1 import ROOT,OUT
from .pilot_scope_v1 import write,sha
from .contracts import AllowedGame,validate_scope,validate_episode,EXPOSURES
from .artifacts import object_hash

def main():
    d=OUT/'data';rows=json.loads((d/'source_records.json').read_text(encoding='utf-8'))
    scope_hash=object_hash([(r['row_id'],r['key'],'fit' if r['outer_pool'] is None else 'outer') for r in rows])
    parsed=[];path=d/'allowed_games_postrun_v1.jsonl'
    with path.open('x',encoding='utf-8',newline='\n') as f:
        for r in rows:
            outer=r['outer_pool'] is not None
            flags={k:None for k in EXPOSURES}
            flags.update(representation_fit=True,normalizer_fit=True,policy_pretrain=False,
                score_seen=True if outer or r['head_part']=='cal' else None,
                error_seen=True if outer else None,protocol_selection=True if outer else None,unknown=True)
            value={'source_row_key':r['key'],'opaque_player_key':r['player'],'target_color':'B' if r['color']==0 else 'W',
                'raw_token':r['raw_hash'],'move_token':r['move_hash'],'canonical_token':r['canonical_hash'],
                'effective_group_token':hashlib.sha256(r['group'].encode()).hexdigest(),
                'role':'outer' if outer else 'fit','rank_train_only':None if outer else r['rank'],
                'exposure':flags,'scope_hash':scope_hash}
            parsed.append(AllowedGame.parse(value));f.write(json.dumps(value,separators=(',',':'))+'\n')
    index=validate_scope(parsed);del parsed
    used=json.loads((d/'allowed_records.json').read_text(encoding='utf-8'))
    plans=json.loads((d/'outer_plans.json').read_text(encoding='utf-8'));checks=0
    for pool in plans:
        supports={p:[used[i]['key'] for i in ids] for p,ids in zip(pool['candidate_order'],pool['supports'])}
        for qn in (5,10,20):
            queries={q['player']:[used[i]['key'] for i in q['indices']] for q in pool['questions'] if q['q']==qn}
            validate_episode(index,pool['candidate_order'],supports,queries,query_count=qn);checks+=1
    write(OUT/'postrun_exposure_and_episode_closure.json',{'status':'passed_postrun_closure_not_backdated',
        'allowed_rows':len(index),'episode_checks':checks,'row_manifest_sha256':sha(path),'scope_hash':scope_hash,
        'exposure_scope':'cumulative conservative historical exposure; current learned-state800 exclusion is independently audited',
        'new_outer_now_score_and_error_exposed':True,'future_pristine_claim_for_this800':False,
        'preflight_boundary':'existing metadata/player/group checks ran before training; formal shared-contract serialization added after scoring',
        'retired400_read':False})
    print('Post-run formal row/episode closure passed',len(index),checks,flush=True)

if __name__=='__main__':main()
