"""Real vectors from exported allowed rows; direct safe model files, no retired bundle loader."""
from __future__ import annotations
import json,time
import numpy as np
from .pilot_scope_v1 import ROOT,RUN,sha,write
from .artifacts import ContractError


def build():
    start=time.perf_counter();d=RUN/'data'
    receipt=json.loads((d/'scope_receipt.json').read_text(encoding='utf-8'))
    if sha(d/'selected_raw_features.npy')!=receipt['selected_raw_features_sha256']:raise ContractError('Exported raw vector digest failed')
    rows=json.loads((d/'allowed_records.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'artifacts/inference_bundle_v1/manifest.json').read_text(encoding='utf-8'))
    bundle_path=ROOT/'artifacts/inference_bundle_v1'
    safe_files=('model_state.pt','fit_transform.npz','feature_schema.json')
    for p in safe_files:
        if sha(bundle_path/p)!=manifest['files'][p]:raise ContractError('Frozen safe model component changed')
    if sha(ROOT/'src/goid/sgf_features.py')!=manifest['parser_sha256']:raise ContractError('Frozen parser changed')
    from goid.fixed_batch import encode_features
    raw=np.load(d/'selected_raw_features.npy',mmap_mode='r',allow_pickle=False)
    (stat,mlp),info=encode_features(raw,{'manifest':manifest,'path':bundle_path},device='cuda')
    from .p1 import compose_game_features
    features=np.lib.format.open_memmap(d/'features.npy',mode='w+',dtype=np.float32,shape=(len(rows),1728))
    for i in range(0,len(rows),2048):
        rr=rows[i:i+2048];count=np.asarray([r['target_moves'] for r in rr],dtype=np.float64)
        stages=np.rint(np.asarray(raw[i:i+len(rr),9:12],dtype=np.float64)*count[:,None])
        if not np.all(stages.sum(axis=1)==count):raise ContractError('Stage counts do not reconstruct original target moves')
        features[i:i+len(rr)]=compose_game_features(stat[i:i+len(rr)],mlp[i:i+len(rr)],raw[i:i+len(rr),:12],
            ['B' if r['color']==0 else 'W' for r in rr],[r['mainline_moves'] for r in rr],stages)
    features.flush();del features
    np.save(d/'stat.npy',stat,allow_pickle=False);np.save(d/'mlp.npy',mlp,allow_pickle=False)
    # Separate physical exports for fit/train, fit-inner calibration and development.
    features=np.load(d/'features.npy',mmap_mode='r',allow_pickle=False)
    for part in ('train','cal','dev'):
        ids=np.asarray([i for i,r in enumerate(rows) if r['part']==part],dtype=np.int64)
        np.save(d/f'{part}_global_indices.npy',ids,allow_pickle=False)
        np.save(d/f'{part}_features.npy',np.array(features[ids],copy=True),allow_pickle=False)
        write(d/f'{part}_records.json',[rows[i] for i in ids])
    hashes={p:sha(d/p) for p in ['features.npy','stat.npy','mlp.npy',*[f'{part}_{suffix}' for part in ('train','cal','dev') for suffix in ('features.npy','records.json','global_indices.npy')]]}
    write(d/'vector_receipt.json',{'source_scope_hash':receipt['selected_scope_hash'],'frozen_encoder':info,
          'safe_model_files':{p:manifest['files'][p] for p in safe_files},'retired_scope_file_loaded':False,
          'hashes':hashes,'elapsed_seconds':time.perf_counter()-start})
    print(json.dumps({'stage':'vectors_complete','rows':len(rows),'elapsed_seconds':time.perf_counter()-start,'GPU':info['gpu_name']},ensure_ascii=False),flush=True)


if __name__=='__main__':build()
