"""Finish an interrupted token replay (tokpilot_20261116_v1.py): the worker has no resume, but its memmaps keep every finished game.
Builds runs/tokpilot_20261116_v1/<name>_rest/ with only the games whose tok_n is still 0 (same patched worker), runs it, then writes
runs/tokpilot_20261116_v1/<name>_merged/{indices,tok_z,tok_a,tok_ply,tok_n,view90}.npy in the ORIGINAL game order.
usage: tokpilot_rest_20261116_v1.py <name> [workers]
"""
from pathlib import Path
import sys, json, shutil, subprocess, hashlib
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; RUN = ROOT / 'runs/tokpilot_20261116_v1'


def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def main():
    name = sys.argv[1]; src = RUN / name; work = RUN / f'{name}_rest'; lines = (src / 'games.jsonl').read_text().splitlines(); ids = np.array([json.loads(s)['index'] for s in lines]); n0 = np.load(src / 'results/tok_n.npy'); todo = np.flatnonzero(n0 == 0); print('games', len(ids), 'missing', len(todo), flush=True)
    if len(todo) and not (work / 'results/RECEIPT.json').exists():
        if work.exists(): shutil.rmtree(work)
        work.mkdir()
        for f in src.iterdir():
            if f.is_file() and f.name not in ('worker.log', 'GPU_CLAIM.json', 'COMPLETION.json'): shutil.copy2(f, work / f.name)
            elif f.is_dir() and f.name != 'results': shutil.copytree(f, work / f.name)
        (work / 'games.jsonl').write_text('\n'.join(lines[i] for i in todo) + '\n'); m = json.loads((work / 'MANIFEST.json').read_text()); m['files']['games.jsonl'] = sha(work / 'games.jsonl'); m['games'] = int(len(todo)); m['rest_of'] = name; (work / 'MANIFEST.json').write_text(json.dumps(m, indent=2))
        with (work / 'worker.log').open('w') as log: rc = subprocess.run([str(ROOT / '.venv_katago_v1/Scripts/python.exe'), 'worker.py'], cwd=work, stdout=log, stderr=subprocess.STDOUT).returncode
        assert rc == 0, f'worker exit {rc}'
    out = RUN / f'{name}_merged'; out.mkdir(exist_ok=False); np.save(out / 'indices.npy', ids)
    for f in ('tok_z', 'tok_a', 'tok_ply', 'tok_n', 'view90'):
        a = np.load(src / f'results/{f}.npy', mmap_mode='r'); o = np.lib.format.open_memmap(out / f'{f}.npy', mode='w+', dtype=a.dtype, shape=a.shape); o[:] = a
        if len(todo):
            r = np.load(work / f'results/{f}.npy', mmap_mode='r'); rid = np.load(work / 'results/indices.npy'); assert np.array_equal(rid, ids[todo]); o[todo] = r
        o.flush()
    n = np.load(out / 'tok_n.npy'); assert n.min() >= 1; print('merged', name, len(n), 'games, mean tokens', float(n.mean()), flush=True)


if __name__ == '__main__':
    main()
