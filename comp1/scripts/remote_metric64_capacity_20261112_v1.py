"""Package only new confirmation SGFs; accept hash-bound owned-job return."""
from pathlib import Path
import sys,json,shutil,tarfile,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
import metric64_capacity_20261112_v1 as e
import numpy as np
p=e.p

SUPERVISOR=r'''from pathlib import Path
import subprocess,os,json,hashlib,tarfile,sys
from datetime import datetime,timezone
r=Path(__file__).resolve().parent
log=(r/'worker.log').open('xb')
p=subprocess.Popen([os.environ.get('AICUP_PYTHON_EXE') or sys.executable,'-B','worker.py'],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
job={'PID':p.pid,'PGID':os.getpgid(p.pid),'start_ticks':Path(f'/proc/{p.pid}/stat').read_text().split()[21],'archive_SHA256':sys.argv[1],'started_utc':datetime.now(timezone.utc).isoformat()}
(r/'REMOTE_JOB.json').write_text(json.dumps(job,indent=2))
code=p.wait();log.close()
f=subprocess.run(['fuser','-v','/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm'],capture_output=True,text=True)
pids=[int(v) for v in f.stdout.split() if v.isdigit()]
users={str(i):Path(f'/proc/{i}/comm').read_text().strip() for i in pids if Path(f'/proc/{i}/comm').exists()}
release={'status':'released' if code==0 else 'failed_released','worker_exit':code,'owned_PID':p.pid,'owned_PID_absent':not Path(f'/proc/{p.pid}').exists(),'GPU_users_after_exit':users,'fuser_stdout':f.stdout,'fuser_stderr':f.stderr,'released_utc':datetime.now(timezone.utc).isoformat()}
(r/'GPU_RELEASE.json').write_text(json.dumps(release,indent=2))
if code==0:
    assert release['owned_PID_absent'] and str(p.pid) not in users
    files=list((r/'results').glob('*'))+[r/x for x in ['GPU_CLAIM.json','GPU_RELEASE.json','REMOTE_JOB.json','COMPLETION.json','worker.log']]
    manifest={str(f.relative_to(r)):hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in files}
    (r/'RETURN_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    with tarfile.open(r/'results_return.tar.gz','x:gz',compresslevel=1) as tf:
        for f in files+[r/'RETURN_MANIFEST.json']:tf.add(f,arcname=str(f.relative_to(r)),recursive=False)
    (r/'RETURN_ARCHIVE.json').write_text(json.dumps({'SHA256':hashlib.file_digest((r/'results_return.tar.gz').open('rb'),'sha256').hexdigest(),'bytes':(r/'results_return.tar.gz').stat().st_size},indent=2))
'''


def package():
    assert p.b.read(e.RUN/'WINNER_FREEZE.json')['winner'];out=e.RUN/'payload_dev';out.mkdir(exist_ok=False);source=e.PRIOR/'feature_payload_v2'
    for f in ('worker.py','model_config.json'):shutil.copy2(source/f,out/f)
    for f in ('vendor','code'):shutil.copytree(source/f,out/f,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    expected=p.b.read(e.PRIOR/'OFFICIAL_WORKER_PARITY.json')['feature_worker_SHA256'];assert p.b.sha(out/'worker.py')==expected
    ids=np.load(e.RUN/'replay_indices.npy');m=p.b.meta();from goid_v4.pilot_scope_v1 import PointReader
    reader=PointReader(p.ROOT/'data/processed/audit.sqlite')
    try:
        with (out/'games.jsonl').open('x',encoding='utf-8',newline='\n') as f:
            for st in range(0,len(ids),256):
                ii=ids[st:st+256];texts={r['row_id']:r['sgf_content'] for r in reader.sample_text(m['rows'][ii])}
                for i in ii:
                    sgf=texts[int(m['rows'][i])];assert hashlib.sha256(sgf.encode()).hexdigest()==m['raw'][i].decode();r={'index':int(i),'sgf':sgf,'raw':m['raw'][i].decode(),'color':'BW'[int(m['colors'][i])],'hd':int(m['hd'][i]),'canonical':m['canonical'][i].decode()};f.write(json.dumps(r)+'\n')
    finally:reader.close()
    with (out/'supervisor.py').open('x',encoding='utf-8',newline='\n') as f:f.write(SUPERVISOR)
    p.b.write(out/'MANIFEST.json',{'files':{f.relative_to(out).as_posix():p.b.sha(f) for f in out.rglob('*') if f.is_file()},'weight_SHA256':p.b.sha(p.n.RUN/'HumanSL_state_only.pt'),'role':'originalsafe_metric64_confirmation_only_no_fit','games':len(ids),'target_labels_packaged':False,'fit_or_optimizer':False,'worker_exact_previous':True})
    archive=e.RUN/'DEV_PAYLOAD.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=3) as tf:
        for f in out.rglob('*'):
            if f.is_file():tf.add(f,arcname=f.relative_to(out).as_posix(),recursive=False)
    r={'archive':str(archive),'SHA256':p.b.sha(archive),'bytes':archive.stat().st_size,'games':len(ids),'worker_SHA256':expected};p.b.write(e.RUN/'DEV_PAYLOAD.json',r);p.b.log(stage='metric64_payload',**r)


def accept(expected,size):
    archive=e.RUN/'DEV_RETURN.tar.gz';assert p.b.sha(archive)==expected and archive.stat().st_size==int(size);out=e.RUN/'dev_return';out.mkdir(exist_ok=False)
    with tarfile.open(archive) as tf:
        assert all((out/m.name).resolve().is_relative_to(out.resolve()) and m.isfile() for m in tf.getmembers());tf.extractall(out,filter='data')
    manifest=p.b.read(out/'RETURN_MANIFEST.json');assert all(p.b.sha(out/f)==h for f,h in manifest.items());assert p.b.read(out/'REMOTE_JOB.json')['archive_SHA256']==p.b.read(e.RUN/'DEV_PAYLOAD.json')['SHA256'];release=p.b.read(out/'GPU_RELEASE.json');assert release['worker_exit']==0 and release['owned_PID_absent']
    dest=e.RUN/'dev_results';dest.mkdir(exist_ok=False)
    for f in (out/'results').iterdir():shutil.copy2(f,dest/f.name)
    for f in out.glob('*.json'):shutil.copy2(f,dest/f.name)
    result={'SHA256':expected,'bytes':int(size),'all_return_hashes_match':True,'input_payload_bound':True,'release':release};p.b.write(e.RUN/'DEV_RETURN_VERIFIED.json',result);p.b.log(stage='metric64_return_verified',**result)


if __name__=='__main__':
    if sys.argv[1]=='package':package()
    else:accept(*sys.argv[2:])
