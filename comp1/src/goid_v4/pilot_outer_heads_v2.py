"""Versioned adapter repair; preserve the failed outer head attempt and exports."""
import json,shutil
from .pilot_outer_v1 import ROOT,OUT
from .pilot_scope_v1 import sha,write
from .pilot_models_v1 import TensorSampler

HEAD=OUT/'heads_v2'

def main():
    HEAD.mkdir(exist_ok=False);d=HEAD/'data';d.mkdir()
    hashes={}
    for part in ('train','cal'):
        records=json.loads((OUT/f'data/{part}_records.json').read_text(encoding='utf-8'))
        for r in records:r['effective_group_token']=r['group']
        if part=='train':TensorSampler(records,31415)
        write(d/f'{part}_records.json',records)
        shutil.copyfile(OUT/f'data/{part}_features.npy',d/f'{part}_features.npy')
        for suffix in ('records.json','features.npy'):
            p=f'{part}_{suffix}';hashes[p]=sha(d/p)
    shutil.copyfile(OUT/'data/scope_receipt.json',d/'scope_receipt.json')
    shutil.copyfile(OUT/'G0_scope.json',HEAD/'G0_scope.json')
    write(d/'vector_receipt.json',{'hashes':hashes,'source_vector_receipt_sha256':sha(OUT/'data/vector_receipt.json'),
        'repair':'effective_group_token aliases existing validated group; values, players, sampling and features unchanged'})
    write(HEAD/'repair_receipt.json',{'failed_attempt_preserved':str(OUT/'models/U0_mean_count_s31415'),
        'failure':'KeyError effective_group_token before first training step',
        'source_records_unchanged':True,'source_trainer_unchanged':True,'outer_scoring_started':False,
        'adapter_sha256':sha(ROOT/'src/goid_v4/pilot_outer_heads_v2.py'),
        'protocol_sha256':sha(ROOT/'configs/research_v4/p1_u0_outer_v1_20261026.json')})
    from . import pilot_train_v1 as engine
    engine.RUN=HEAD
    for seed in (31415,27182,16180):engine.train('U0_mean_count',seed)

if __name__=='__main__':main()
