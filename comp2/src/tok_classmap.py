"""Map the classes of the competition-1 token nets (AM-softmax over training-file owners; the training files of both competitions are
byte-identical) to comp2 account names (raw player_id).  Route: comp1 labels (game index -> class) -> comp1 TRAINING-bank payload
game lists (index -> sha256 of the SGF text, field 'raw') -> comp2 training CSVs (sha256(sgf_content) -> player_id).
Only training-bank payloads are read (bigbank*/payload_c*, bigbank6 payloads); no comp1 test / official / dev payloads.
usage: tok_classmap.py COMP1_LABELS.npz COMP2_TRAIN_DIR OUT.npz PAYLOAD_GLOB [PAYLOAD_GLOB ...]
"""
import sys, glob, json, hashlib, csv, collections, numpy as np
csv.field_size_limit(10**9)
L = np.load(sys.argv[1]); ids = L["ids"].astype(np.int64); y = L["y"].astype(np.int64); want = set(ids.tolist())
raw = {}
files = sorted(set(f for g in sys.argv[4:] for f in glob.glob(g)))
for f in files:
    assert "official" not in f, f      # bigbank90 payload_dev = comp1's training-player bank (only LABELLED training ids are read)
    n = 0
    for ln in open(f, encoding="utf-8"):
        i = ln.find('"index": '); j = ln.find(",", i); idx = int(ln[i + 9:j])
        if idx in want and idx not in raw:
            k = ln.find('"raw": "'); raw[idx] = ln[k + 8:k + 72]; n += 1
    print(f, "new", n, flush=True)
print("labelled ids", len(want), "with raw hash", len(raw), flush=True)
h2p = {}
for f in sorted(glob.glob(sys.argv[2] + "/train_*.csv")):
    for r in csv.DictReader(open(f, encoding="utf-8", newline="")):
        h2p[hashlib.sha256(r["sgf_content"].encode()).hexdigest()] = r["player_id"]
    print(f, len(h2p), flush=True)
votes = collections.defaultdict(collections.Counter); miss = 0
for i, c in zip(ids, y):
    r = raw.get(int(i))
    if r is None: continue
    p = h2p.get(r)
    if p is None: miss += 1; continue
    votes[int(c)][p] += 1
C = int(y.max()) + 1; names = np.array([""] * C, dtype=object); purity = np.zeros(C)
for c in range(C):
    if votes[c]: p, n = votes[c].most_common(1)[0]; names[c] = p; purity[c] = n / sum(votes[c].values())
print("classes", C, "mapped", int((names != "").sum()), "hash misses", miss, "purity min", purity[names != ""].min(), "mean", purity[names != ""].mean())
np.savez(sys.argv[3], names=names.astype(str), purity=purity, n=np.array([sum(votes[c].values()) for c in range(C)]))
