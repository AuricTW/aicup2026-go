"""Runs ON THE 4090 VM: full-scale token replay (see tokpilot_20261116_v1.py / tokhead_20261116_v1.py for the pilot:
learned pooling over per-decision HumanSL tokens 0.573 vs 0.537 for the aggregate heads on the same 18.9k games, 0.618 together).
For every source payload dir (already on the VM from the view90 replays) the frozen worker is copied, its GPU-exclusivity guards are
relaxed (shared card), and the four token memmaps are added; selection / encoding / forward are untouched, view90 is produced again and
can be compared with the accepted one.
usage (in <vm_runs_dir>, see configs/paths.example.yaml): python tokfull_vm_20261116_v1.py <stream name> <src dir> [<src dir> ...]
writes tokfull_20261116/<src dir>/results/{tok_z,tok_a,tok_ply,tok_n,view90,indices}.npy ; finished dirs are skipped (TOKENS.json).
"""
import sys, os, re, json, shutil, subprocess, hashlib, time
from pathlib import Path
import numpy as np
import comp1_paths as CP
BASE = CP.path('vm_runs_dir', 'runs'); OUT = BASE / 'tokfull_20261116'; PY = CP.get('python_exe') or sys.executable


def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def patch(t):
    must = [("    return original8,p.aggregate(z[middle],a[middle]),p.aggregate(z,a)", "    return original8,p.aggregate(z[middle],a[middle]),p.aggregate(z,a),z.astype(np.float16),a.astype(np.float32),np.asarray(plies,np.int16)"),
            (" probe=[encode64_record(item) for item in tasks[:4]]", " TK={'z':np.lib.format.open_memmap(dest/'tok_z.npy',mode='w+',dtype=np.float16,shape=(len(ids),90,1152)),'a':np.lib.format.open_memmap(dest/'tok_a.npy',mode='w+',dtype=np.float32,shape=(len(ids),90,10)),'ply':np.lib.format.open_memmap(dest/'tok_ply.npy',mode='w+',dtype=np.int16,shape=(len(ids),90)),'n':np.lib.format.open_memmap(dest/'tok_n.npy',mode='w+',dtype=np.uint8,shape=(len(ids),))}\n probe=[encode64_record(item) for item in tasks[:4]]"),
            ("     for key,value in zip((8,64,90),values):arrays[key][j]=value", "     for key,value in zip((8,64,90),values):arrays[key][j]=value\n     k_=len(values[3]);assert values[4].shape==(k_,10) and len(values[5])==k_;TK['z'][j,:k_]=values[3];TK['a'][j,:k_]=values[4];TK['ply'][j,:k_]=values[5];TK['n'][j]=k_"),
            (" write(dest/'FALLBACKS.json',fallbacks)", " [v_.flush() for v_ in TK.values()]\n write(dest/'FALLBACKS.json',fallbacks)")]
    for a, b in must: assert t.count(a) == 1, (t.count(a), a[:60]); t = t.replace(a, b)
    t, n1 = re.subn(r"assert users and all\(x=='Xorg' for x in users\.values\(\)\),users", "pass", t)      # other jobs share the card
    t, n2 = re.subn(r"free,total=torch\.cuda\.mem_get_info\(\);assert free>\d+\*2\*\*30", "free,total=torch.cuda.mem_get_info();assert free>3*2**30", t)
    t, n3 = re.subn(r"assert time\.perf_counter\(\)-start<\d+", "assert time.perf_counter()-start<400000", t); t = t.replace("torch.cuda.set_per_process_memory_fraction(.65)", "pass"); print('guard patches', n1, n2, n3, flush=True); assert n3 == 1
    return t


def main():
    stream = sys.argv[1]; OUT.mkdir(exist_ok=True)
    for name in sys.argv[2:]:
        src = BASE / name; work = OUT / name
        if (work / 'TOKENS.json').exists(): print(stream, name, 'already done', flush=True); continue
        if work.exists(): shutil.rmtree(work)
        work.mkdir()
        for f in ('basis64.py', 'games.jsonl', 'MANIFEST.json', 'model_config.json', 'worker.py', 'supervisor.py'): shutil.copy2(src / f, work / f)
        for d in ('code', 'vendor'): shutil.copytree(src / d, work / d)
        w = work / 'worker.py'; w.write_text(patch(w.read_text())); m = json.loads((work / 'MANIFEST.json').read_text()); m['files']['worker.py'] = sha(w); m['token_replay'] = {'source': str(src), 'stream': stream}; (work / 'MANIFEST.json').write_text(json.dumps(m, indent=2)); t0 = time.time()
        with (work / 'worker.log').open('w') as log: rc = subprocess.run([PY, 'worker.py'], cwd=work, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0: print(stream, name, 'FAILED rc', rc, flush=True); continue
        res = work / 'results'; n = np.load(res / 'tok_n.npy'); old = src / 'results/view90.npy'; cmp = None
        if old.exists() and np.array_equal(np.load(src / 'results/indices.npy'), np.load(res / 'indices.npy')): a = np.load(old, mmap_mode='r'); b = np.load(res / 'view90.npy', mmap_mode='r'); cmp = float(np.abs(a[:3000] - b[:3000]).max())
        for f in ('view8.npy', 'view64.npy'): (res / f).unlink()
        (work / 'TOKENS.json').write_text(json.dumps({'games': int(len(n)), 'min_tokens': int(n.min()), 'mean_tokens': float(n.mean()), 'wall': time.time() - t0, 'view90_maxabs_vs_accepted_first3000': cmp}, indent=1)); print(stream, name, 'done', len(n), f'{time.time() - t0:.0f}s', 'view90 diff', cmp, flush=True)


if __name__ == '__main__':
    main()
