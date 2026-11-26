"""Seal only new experiment roots and explicit new code/configs; no old data walk."""
import json,platform,sys,subprocess
from datetime import datetime
from pathlib import Path
from .pilot_scope_v1 import ROOT,RUN,sha,write
from .pilot_outer_v1 import OUT

def main():
    packages=json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True))
    write(OUT/'environment_packages.json',{'python':sys.version,'platform':platform.platform(),'packages':packages,'network_required':False})
    protected=json.loads((RUN/'protected_text_before.json').read_text(encoding='utf-8'))['files']
    changed=[r['path'] for r in protected if sha(ROOT/r['path'])!=r['sha256']]
    if changed:raise RuntimeError('Protected source changed: '+str(changed))
    paths=set()
    for root in (RUN,OUT,ROOT/'docs/research_v4/p1_pilot_v1_20261026',ROOT/'docs/research_v4/p1_u0_outer_v1_20261026'):
        for p in root.rglob('*'):
            if p.is_symlink():raise RuntimeError('Unexpected symlink in bounded new roots')
            if p.is_file():paths.add(p)
    paths.update((ROOT/'src/goid_v4').glob('pilot_*.py'))
    paths.update((ROOT/'tests/research_v4/pilot_v1').glob('*.py'))
    for p in ('configs/research_v4/p1_pilot_v1_20261026.json','configs/research_v4/p1_u0_outer_v1_20261026.json',
              'docs/research_v4/EXPERIMENT_STATUS_v1_20261026.json'):paths.add(ROOT/p)
    excluded={'ARTIFACT_MANIFEST.json','SEAL_VALIDATION.json','console_seal.log'}
    records=[]
    for p in sorted(paths):
        if p.name in excluded or '__pycache__' in p.parts:continue
        records.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    manifest=OUT/'ARTIFACT_MANIFEST.json'
    write(manifest,{'schema':'bounded_new_experiment_manifest_v1','created_at':datetime.now().astimezone().isoformat(),
        'excluded_self_and_live_seal_files':sorted(excluded),'files':records})
    loaded=json.loads(manifest.read_text(encoding='utf-8'));seen=set()
    for row in loaded['files']:
        if row['path'] in seen or not (ROOT/row['path']).is_file() or (ROOT/row['path']).stat().st_size!=row['bytes']:
            raise RuntimeError('Manifest index validation failed')
        seen.add(row['path'])
    result={'status':'sealed','files':len(records),'bytes':sum(r['bytes'] for r in records),
        'manifest_sha256':sha(manifest),'manifest_unique_paths_and_sizes_verified':True,
        'hashes_computed_by_streaming_each_selected_file_once':True,'second_content_hash_pass':False,
        'protected_existing_text_files_rechecked':len(protected),'protected_changed':[],
        'original_mixed_data_or_retired_files_hashed_or_walked':False,
        'scope':'new versioned runs/docs/code only; not a portable ZIP or whole-project snapshot'}
    write(OUT/'SEAL_VALIDATION.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
