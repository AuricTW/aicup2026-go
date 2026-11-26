"""Token pilot: replay an existing view90 payload locally and ALSO keep the per-decision HumanSL tokens.
Why: every head so far sees one 2339-d vector per game = fixed statistics (mean / std) of <= 90 per-decision vectors
(local 384, board-mean 384, local minus policy expectation 384, + 10 scalars).  Heads on that aggregate are saturated (full bank,
episodic loss, 10 seeds: +0.012 in total) and the colour mix is where the score is lost.  With the tokens a learned pooling / small
Transformer can decide which decisions matter.  This script only produces the data for a pilot.
The frozen worker of the payload is copied; patches = the platform guards of local_replay_20261115_v1.py + four token memmaps:
  tok_z.npy  [N, 90, 1152] float16   tok_a.npy [N, 90, 10] float32   tok_ply.npy [N, 90] int16   tok_n.npy [N] uint8
(the float16 token values are exactly the ones the worker aggregates into view90).  Selection, encoding, forward: untouched.
usage: tokpilot_20261116_v1.py <payload dir relative to runs/> <name> [workers]   ->  runs/tokpilot_20261116_v1/<name>/results/
"""
from pathlib import Path
import sys, json, shutil, subprocess, hashlib, time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); import comp1_paths as CP  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]; WEIGHT = ROOT / 'runs/fisher_policy_pilot_20261105_v2/assets/HumanSL_state_only.pt'; RUN = ROOT / 'runs/tokpilot_20261116_v1'


def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def main():
    src = ROOT / 'runs' / sys.argv[1]; name = sys.argv[2]; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8; work = RUN / name; RUN.mkdir(exist_ok=True); manifest = json.loads((src / 'MANIFEST.json').read_text()); assert sha(WEIGHT) == manifest['weight_SHA256']
    work.mkdir(exist_ok=False)
    for f in src.iterdir():
        if f.is_file(): shutil.copy2(f, work / f.name)
        elif f.name != 'results': shutil.copytree(f, work / f.name)
    w = work / 'worker.py'; t = w.read_text(encoding='utf-8')
    reps = [("Path(os.environ.get('AICUP_HUMANSL_WEIGHT') or 'HumanSL_state_only.pt')", f'Path({str(WEIGHT)!r})'),
            ("check=subprocess.run(['fuser','-v','/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm'],capture_output=True,text=True)", "users={'0':'local'}"),
            ("assert not any(x in check.stderr.lower() for x in ['permission denied','cannot stat'])", "pass"),
            ("pids=set(int(x) for x in check.stdout.split() if x.isdigit());users={str(pid):Path(f'/proc/{pid}/comm').read_text().strip() for pid in pids};assert users and all(x=='Xorg' for x in users.values()),users", "pass"),
            ("assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name();free,total=torch.cuda.mem_get_info();assert free>18*2**30", "assert torch.cuda.is_available();free,total=torch.cuda.mem_get_info();assert free>3*2**30"),
            ("cores=os.cpu_count();load=os.getloadavg()[0];workers=8 if cores>=12 and load<cores*.5 else 4", f"cores=os.cpu_count();load=0.0;workers={workers}"),
            ("'PGID':os.getpgid(0),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],", "'PGID':0,'start_ticks':'local',"),
            ("assert time.perf_counter()-start<5400", "assert time.perf_counter()-start<90000"),
            ("torch.cuda.set_per_process_memory_fraction(.65)", "pass"),
            # tokens
            ("    return original8,p.aggregate(z[middle],a[middle]),p.aggregate(z,a)", "    return original8,p.aggregate(z[middle],a[middle]),p.aggregate(z,a),z.astype(np.float16),a.astype(np.float32),np.asarray(plies,np.int16)"),
            (" probe=[encode64_record(item) for item in tasks[:4]]", " TK={'z':np.lib.format.open_memmap(dest/'tok_z.npy',mode='w+',dtype=np.float16,shape=(len(ids),90,1152)),'a':np.lib.format.open_memmap(dest/'tok_a.npy',mode='w+',dtype=np.float32,shape=(len(ids),90,10)),'ply':np.lib.format.open_memmap(dest/'tok_ply.npy',mode='w+',dtype=np.int16,shape=(len(ids),90)),'n':np.lib.format.open_memmap(dest/'tok_n.npy',mode='w+',dtype=np.uint8,shape=(len(ids),))}\n probe=[encode64_record(item) for item in tasks[:4]]"),
            ("     for key,value in zip((8,64,90),values):arrays[key][j]=value", "     for key,value in zip((8,64,90),values):arrays[key][j]=value\n     k_=len(values[3]);assert values[4].shape==(k_,10) and len(values[5])==k_;TK['z'][j,:k_]=values[3];TK['a'][j,:k_]=values[4];TK['ply'][j,:k_]=values[5];TK['n'][j]=k_"),
            (" write(dest/'FALLBACKS.json',fallbacks)", " [v_.flush() for v_ in TK.values()]\n write(dest/'FALLBACKS.json',fallbacks)")]
    for a, b in reps: assert t.count(a) == 1, (t.count(a), a[:80]); t = t.replace(a, b)
    w.write_text(t, encoding='utf-8'); manifest['files']['worker.py'] = sha(w); manifest['token_pilot'] = {'source_payload': str(src), 'patched_worker_SHA256': sha(w), 'change': 'local platform guards + token memmaps; compute unchanged', 'workers': workers}
    (work / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2)); t0 = time.time()
    with (work / 'worker.log').open('w') as log: rc = subprocess.run([CP.get('python_exe') or sys.executable, 'worker.py'], cwd=work, stdout=log, stderr=subprocess.STDOUT).returncode
    assert rc == 0, f'worker exit {rc}'; res = work / 'results'; rec = json.loads((res / 'RECEIPT.json').read_text()); assert rec['all_valid'] and rec['games'] == manifest['games']; n = np.load(res / 'tok_n.npy'); assert n.min() >= 1
    (work / 'TOKENS.json').write_text(json.dumps({'games': int(len(n)), 'mean_tokens': float(n.mean()), 'seconds': rec['seconds'], 'wall': time.time() - t0, 'view90_SHA256': rec['SHA256']['view90.npy']}, indent=1)); print(name, 'done', len(n), 'games', f"{rec['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
