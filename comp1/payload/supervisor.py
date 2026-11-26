from pathlib import Path
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
