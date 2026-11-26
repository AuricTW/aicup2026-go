"""CSV -> replayed SGF features -> frozen encoders -> original/color RRF.

No audit database, labels, existing feature matrix or experiment outputs are
used by this runtime. The local bundle contains inference-only weights,
fit statistics and a retired-cohort exclusion hash list. Feature caching is
content-addressed and checked; all outputs are immutable local files.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import sys
import time
import traceback

import numpy as np

from goid.sgf_features import analyze_sgf, FEATURE_DIM

ROOT = Path(__file__).resolve().parents[2]
HEADER = ["question_id", "top_1", "top_2", "top_3", "top_4", "top_5"]
QUERY_HEADER = ["question_id", "num_games"] + [f"sgf_{i}" for i in range(1,21)] + [f"color_{i}" for i in range(1,21)]
CANDIDATE_HEADER = ["game_id", "player_id", "sgf_content", "color"]


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def utc(): return datetime.now(timezone.utc).isoformat()


def identity(value,description):
    if not isinstance(value,str) or not value or value!=value.strip():
        raise ValueError(f"{description}: empty or padded identifier")
    return value


def rows_csv(path,required):
    csv.field_size_limit(16*1024*1024)
    with Path(path).open("r",encoding="utf-8-sig",newline="") as stream:
        reader=csv.reader(stream,strict=True)
        header=next(reader,None)
        if not header or len(header)!=len(set(header)) or any(not x or x!=x.strip() for x in header):
            raise ValueError(f"{path}: invalid or duplicated header")
        if not set(required)<=set(header):
            raise ValueError(f"{path}: missing columns {sorted(set(required)-set(header))}")
        for row in reader:
            if len(row)!=len(header):
                raise ValueError(f"{path}:{reader.line_num}: wrong column count")
            yield reader.line_num,dict(zip(header,row))


@dataclass(frozen=True)
class Game:
    key: str
    raw_hash: str
    color: str
    sgf: str


@dataclass
class Pool:
    name: str
    candidate_ids: list
    supports: dict
    questions: list
    source: dict


def feature_key(raw_hash,color,parser_sha,sgfmill_version):
    return hashlib.sha256(f"goid-feature-cache-v1|{parser_sha}|{sgfmill_version}|{raw_hash}|{color}".encode()).hexdigest()


def add_game(games,sgf,color,bundle):
    if color not in ("B","W") or not sgf or not sgf.strip():
        raise ValueError("Every declared game requires nonempty SGF and color B/W")
    raw=hashlib.sha256(sgf.encode("utf-8")).hexdigest()
    if raw in bundle["banned_raw"]:
        raise ValueError("Retired-cohort SGF hash rejected before feature extraction")
    key=feature_key(raw,color,bundle["manifest"]["parser_sha256"],bundle["manifest"]["runtime"]["sgfmill"])
    item=Game(key,raw,color,sgf)
    previous=games.setdefault(key,item)
    if previous!=item:
        raise ValueError("Feature-cache key collision")
    return key


def load_pool(spec,games,bundle):
    name=identity(spec["name"],"pool name")
    if Path(name).name!=name or any(x in name for x in "/\\:"):
        raise ValueError("Unsafe pool name")
    candidates=Path(spec["candidates_csv"]).resolve(); queries=Path(spec["queries_csv"]).resolve()
    source={"candidates_csv":str(candidates),"queries_csv":str(queries),"candidates_sha256":sha(candidates),"queries_sha256":sha(queries)}
    supports={}; game_ids=set()
    for line,row in rows_csv(candidates,CANDIDATE_HEADER):
        p=identity(row["player_id"],f"candidate at {line}")
        if p in bundle["banned_players"]:
            raise ValueError("Retired player ID rejected before feature extraction")
        gid=identity(row["game_id"],"game_id")
        if (p,gid) in game_ids:
            raise ValueError("Repeated candidate player/game_id")
        game_ids.add((p,gid))
        key=add_game(games,row["sgf_content"],row["color"],bundle)
        support=supports.setdefault(p,[])
        if key in support:
            raise ValueError("Repeated SGF/color inside one candidate")
        support.append(key)
    expected=spec.get("expected_candidates",400); references=spec.get("reference_games",100)
    if isinstance(expected,bool) or not isinstance(expected,int) or not 5<=expected<=65535 or len(supports)!=expected:
        raise ValueError("Wrong expected/actual candidate count")
    if not isinstance(references,int) or isinstance(references,bool) or references<1 or any(len(ids)!=references for ids in supports.values()):
        raise ValueError("Every candidate must have the declared support-game count")
    questions=[]; seen=set()
    for line,row in rows_csv(queries,QUERY_HEADER):
        qid=identity(row["question_id"],f"question at {line}")
        if qid in seen:
            raise ValueError("Repeated question ID")
        seen.add(qid)
        try: count=int(row["num_games"])
        except ValueError: raise ValueError("Invalid num_games") from None
        if str(count)!=row["num_games"] or not 5<=count<=20:
            raise ValueError("num_games must be an integer from5 through20")
        keys=[]
        for i in range(1,21):
            if i<=count:
                keys.append(add_game(games,row[f"sgf_{i}"],row[f"color_{i}"],bundle))
            elif row[f"sgf_{i}"] or row[f"color_{i}"]:
                raise ValueError("Nonempty trailing SGF/color beyond num_games")
        if len(set(keys))!=count:
            raise ValueError("Repeated SGF/color inside one question")
        questions.append({"question_id":qid,"keys":keys,"num_games":count})
    expected_questions=spec.get("expected_queries",400)
    if isinstance(expected_questions,bool) or not isinstance(expected_questions,int) or expected_questions<1 or not questions or len(questions)!=expected_questions:
        raise ValueError("Wrong expected/actual question count")
    return Pool(name,list(supports),supports,questions,source)


def load_bundle(path,allow_synthetic_cpu=False):
    path=Path(path).resolve(); manifest=read_json(path/"manifest.json")
    if manifest.get("schema")!="goid-inference-bundle-v1":
        raise ValueError("Unsupported model bundle schema")
    if read_json(path/"manifest.sha256.json")["manifest.json"]!=sha(path/"manifest.json"):
        raise ValueError("Bundle manifest checksum failed")
    if manifest.get("synthetic_control") and not allow_synthetic_cpu:
        raise ValueError("Synthetic bundle cannot be used as competition evidence")
    for rel,digest in manifest["files"].items():
        target=(path/rel).resolve()
        if not target.is_relative_to(path) or sha(target)!=digest:
            raise ValueError(f"Model bundle checksum failed: {rel}")
    for rel,digest in manifest["code_hashes"].items():
        if sha(ROOT/rel)!=digest:
            raise ValueError(f"Pinned inference source changed: {rel}")
    for package in ("numpy","torch","sgfmill"):
        if importlib.metadata.version(package)!=manifest["runtime"][package]:
            raise ValueError(f"Pinned runtime version changed: {package}")
    if manifest["feature_dimension"]!=FEATURE_DIM or manifest["parser_sha256"]!=sha(ROOT/"src/goid/sgf_features.py"):
        raise ValueError("Bundle feature/parser version differs")
    with np.load(path/"retired_scope.npz",allow_pickle=False) as excluded:
        banned_players=set(excluded["player_ids"].tolist())
        banned_raw=set(excluded["raw_hashes"].tolist())
        banned_canonical=set(excluded["canonical_hashes"].tolist())
    return {"path":path,"manifest":manifest,"manifest_sha256":sha(path/"manifest.json"),"banned_players":banned_players,"banned_raw":banned_raw,"banned_canonical":banned_canonical}


def parse_batch(batch):
    results=[]
    for item in batch:
        try:
            parsed=analyze_sgf(item.sgf,item.color,include_features=True)
            if parsed["board_size"]!=19:
                raise ValueError("Frozen model accepts19x19 SGF only")
            vector=np.asarray(parsed["features"],np.float32)
            if vector.shape!=(FEATURE_DIM,) or not np.isfinite(vector).all():
                raise ValueError("Invalid SGF feature vector")
            results.append((item.key,vector.tobytes(),parsed["canonical_hash"],None))
        except Exception as error:
            results.append((item.key,None,None,f"{type(error).__name__}: {error}"))
    return results


class FeatureCache:
    def __init__(self,path):
        self.path=Path(path).resolve(); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS features_v1 (key TEXT PRIMARY KEY,raw_hash TEXT NOT NULL,color TEXT NOT NULL,canonical_hash TEXT NOT NULL,vector_sha256 TEXT NOT NULL,vector BLOB NOT NULL)")
        self.db.commit()

    def close(self): self.db.close()

    def fetch(self,keys):
        for start in range(0,len(keys),800):
            batch=keys[start:start+800]
            yield from self.db.execute(f"SELECT key,raw_hash,color,canonical_hash,vector_sha256,vector FROM features_v1 WHERE key IN ({','.join('?' for _ in batch)})",batch)

    @staticmethod
    def check(row,game,banned_canonical):
        key,raw,color,canonical,digest,blob=row
        if raw!=game.raw_hash or color!=game.color or len(blob)!=FEATURE_DIM*4 or hashlib.sha256(blob).hexdigest()!=digest:
            raise ValueError("Feature cache identity/content checksum failed")
        if canonical in banned_canonical:
            raise ValueError("Retired D4 canonical hash rejected before model encoding")
        vector=np.frombuffer(blob,np.float32)
        if not np.isfinite(vector).all(): raise ValueError("Nonfinite cached feature")
        return vector


def extract_features(games,bundle,cache_path,workers,output):
    if not 1<=workers<=8: raise ValueError("workers must be1..8")
    keys=sorted(games); positions={k:i for i,k in enumerate(keys)}
    matrix=np.empty((len(keys),FEATURE_DIM),np.float32)
    cache=FeatureCache(cache_path); found=set(); index={}; start=time.perf_counter()
    try:
        for row in cache.fetch(keys):
            key=row[0]; matrix[positions[key]]=cache.check(row,games[key],bundle["banned_canonical"])
            found.add(key); index[key]={"key":key,"raw_hash":row[1],"color":row[2],"canonical_hash":row[3],"feature_sha256":row[4]}
        missing=[games[k] for k in keys if k not in found]
        parsed=0
        def accept(batch):
            nonlocal parsed
            insert=[]
            for key,blob,canonical,error in batch:
                if error:
                    with (output/"parse_errors.jsonl").open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps({"key":key,"raw_hash":games[key].raw_hash,"error":error})+"\n")
                    raise ValueError("SGF parse failed; no games/questions are silently dropped")
                digest=hashlib.sha256(blob).hexdigest(); item=games[key]
                row=(key,item.raw_hash,item.color,canonical,digest,blob)
                matrix[positions[key]]=cache.check(row,item,bundle["banned_canonical"])
                insert.append(row); index[key]={"key":key,"raw_hash":item.raw_hash,"color":item.color,"canonical_hash":canonical,"feature_sha256":digest}
            cache.db.executemany("INSERT INTO features_v1 VALUES (?,?,?,?,?,?)",insert); cache.db.commit()
            parsed+=len(batch)
            if parsed%1024==0 or parsed==len(missing):
                print(json.dumps({"stage":"sgf_parse","parsed":parsed,"cache_misses":len(missing),"elapsed_seconds":time.perf_counter()-start}),flush=True)
        batches=iter([missing[i:i+128] for i in range(0,len(missing),128)])
        if workers==1:
            for batch in batches: accept(parse_batch(batch))
        elif missing:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as executor:
                pending={}
                for _ in range(workers*2):
                    batch=next(batches,None)
                    if batch is not None: pending[executor.submit(parse_batch,batch)]=True
                while pending:
                    done,_=wait(pending,return_when=FIRST_COMPLETED)
                    for future in done:
                        del pending[future]; accept(future.result())
                        batch=next(batches,None)
                        if batch is not None: pending[executor.submit(parse_batch,batch)]=True
        with (output/"feature_index.jsonl").open("w",encoding="utf-8") as stream:
            for key in keys: stream.write(json.dumps(index[key])+"\n")
        write_json(output/"feature_keys.json",keys)
        stats={"unique_games":len(keys),"cache_hits":len(found),"cache_misses":len(missing),"actually_parsed":parsed,"workers":workers,"elapsed_seconds":time.perf_counter()-start,"cache":str(Path(cache_path).resolve()),"cache_key_includes_parser_color_raw_hash":True}
        write_json(output/"cache_report.json",stats)
        return matrix,keys,stats
    finally:
        cache.close()


def normalize_rows(matrix):
    return matrix/np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),np.float32(1e-12))


def statistical_vectors(features,bundle):
    with np.load(bundle["path"]/"fit_transform.npz",allow_pickle=False) as fit:
        indices=fit["selected_indices"]; mean=fit["mean"]; std=fit["std"]
    schema=read_json(bundle["path"]/"feature_schema.json")
    # Match the original transform's C-order copy before reductions exactly.
    value=np.asarray(features[:,indices],dtype=np.float32).copy(); offset=0
    for group in schema["groups"]:
        if group["kind"]!="histogram": continue
        width=group["stop"]-group["start"]; hist=value[:,offset:offset+width]
        hist/=np.maximum(hist.sum(axis=1,keepdims=True),np.float32(1e-12)); offset+=width
    if offset!=len(indices) or np.any(std<=0) or not np.isfinite(value).all():
        raise ValueError("Invalid frozen statistical transform")
    return normalize_rows((value-mean)/std)


def encode_features(features,bundle,device="cuda",batch_size=1024):
    import torch
    from goid.feature_metric import FeatureMetric,FrozenFeaturePreprocessor
    synthetic=bundle["manifest"].get("synthetic_control",False)
    if device!="cuda" and not (synthetic and device=="cpu"):
        raise ValueError("Real inference requires local RTX3060 CUDA; CPU is synthetic-only")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if device=="cuda" and (not torch.cuda.is_available() or "RTX 3060" not in torch.cuda.get_device_name()):
        raise RuntimeError("Local RTX3060 CUDA unavailable")
    config=bundle["manifest"]["model_config"]
    pre=FrozenFeaturePreprocessor(bundle["path"],"standardized",10)
    model=FeatureMetric(len(pre.indices),config["embedding_dim"],config["hidden_dim"],config["dropout"]).to(device)
    state=torch.load(bundle["path"]/"model_state.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(state,strict=True); model.eval().requires_grad_(False)
    def digest():
        h=hashlib.sha256()
        for name,t in sorted(model.state_dict().items()):
            h.update(name.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=digest()
    statistical=np.empty((len(features),len(pre.indices)),np.float32)
    learned=np.empty((len(features),config["embedding_dim"]),np.float32)
    start=time.perf_counter()
    with torch.inference_mode(),torch.autocast(device_type=device,enabled=False):
        for i in range(0,len(features),batch_size):
            batch=features[i:i+batch_size]
            statistical[i:i+batch_size]=statistical_vectors(batch,bundle)
            learned[i:i+batch_size]=model(torch.from_numpy(pre(batch)).to(device)).float().cpu().numpy()
    after=digest()
    if before!=after or any(p.grad is not None for p in model.parameters()) or not np.isfinite(learned).all():
        raise ValueError("Model mutated, gradients produced or nonfinite embedding")
    return (statistical,learned),{"device":device,"gpu_name":torch.cuda.get_device_name() if device=="cuda" else None,"precision":"fp32","autocast":False,"tf32":False,"model_before":before,"model_after":after,"unchanged":True,"optimizer_constructed":False,"backward_calls":0,"fit_calls":0,"torch":torch.__version__,"cuda":torch.version.cuda,"numpy":np.__version__,"elapsed_seconds":time.perf_counter()-start}


def component_scores(vectors,pool,games,positions):
    def profile(keys): return normalize_rows(vectors[[positions[k] for k in keys]].mean(axis=0,keepdims=True))[0]
    d=vectors.shape[1]; support=np.empty((len(pool.candidate_ids),3,d),np.float32)
    counts=np.zeros((len(pool.candidate_ids),2),np.int64); fallback=np.zeros_like(counts,bool)
    for i,p in enumerate(pool.candidate_ids):
        keys=pool.supports[p]; support[i,0]=profile(keys)
        for c,color in enumerate(("B","W")):
            selected=[k for k in keys if games[k].color==color]; counts[i,c]=len(selected)
            missing=not selected or np.linalg.norm(vectors[[positions[k] for k in selected]].mean(axis=0))<=1e-12
            fallback[i,c]=missing; support[i,c+1]=support[i,0] if missing else profile(selected)
    qglobal=[]; qcolors=np.zeros((len(pool.questions),2,d),np.float32); qcounts=np.zeros((len(pool.questions),2),np.int64)
    for i,q in enumerate(pool.questions):
        keys=q["keys"]; qglobal.append(profile(keys))
        for c,color in enumerate(("B","W")):
            selected=[k for k in keys if games[k].color==color]; qcounts[i,c]=len(selected)
            if selected: qcolors[i,c]=profile(selected)
    weights=qcounts.astype(np.float32)/qcounts.sum(axis=1,keepdims=True).astype(np.float32)
    original=np.stack(qglobal)@support[:,0].T
    color=weights[:,:1]*(qcolors[:,0]@support[:,1].T)+weights[:,1:]*(qcolors[:,1]@support[:,2].T)
    return original,color,{"support_color_counts":counts,"support_fallback":fallback,"query_color_counts":qcounts,"query_color_weights":weights}


def stable_ranks(scores):
    if scores.ndim!=2 or not np.isfinite(scores).all(): raise ValueError("Scores must be finite")
    return np.argsort(np.argsort(-scores,axis=1,kind="stable"),axis=1,kind="stable")+1


def rrf(a,b): return .5/(60+stable_ranks(a))+.5/(60+stable_ranks(b))


def choose_scores(color_a,color_b,original,method):
    if method not in ("color","original"): raise ValueError("Unknown inference method")
    if not np.isfinite(original).all(): raise ValueError("Original RRF unusable")
    good=np.isfinite(color_a).all(axis=1)&np.isfinite(color_b).all(axis=1)
    color=original.copy()
    if good.any(): color[good]=rrf(color_a[good],color_b[good])
    return color,(color if method=="color" else original),~good


def write_top5(path,pool,scores):
    if scores.shape!=(len(pool.questions),len(pool.candidate_ids)) or not np.isfinite(scores).all():
        raise ValueError("Wrong score matrix for CSV")
    order=np.argsort(-scores,axis=1,kind="stable")[:,:5]
    with Path(path).open("w",encoding="utf-8",newline="") as stream:
        writer=csv.writer(stream,lineterminator="\n"); writer.writerow(HEADER)
        for q,ranking in zip(pool.questions,order): writer.writerow([q["question_id"]]+[pool.candidate_ids[j] for j in ranking])


def check_output(path,pools):
    owners={q["question_id"]:set(p.candidate_ids) for p in pools for q in p.questions}; seen=set()
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.reader(stream,strict=True)
        if next(reader,None)!=HEADER: raise ValueError("Submission header differs")
        for row in reader:
            if len(row)!=6 or row[0] not in owners or row[0] in seen or len(set(row[1:]))!=5 or not set(row[1:])<=owners[row[0]]:
                raise ValueError("Submission question, duplicates or pool membership invalid")
            seen.add(row[0])
    if seen!=set(owners): raise ValueError("Incomplete submission question set")
    return {"valid":True,"rows":len(seen),"scope":"Header, exact question set, five distinct IDs and per-question pool membership; no platform submission"}


def combine_files(paths,target):
    with Path(target).open("w",encoding="utf-8",newline="") as stream:
        writer=csv.writer(stream,lineterminator="\n"); writer.writerow(HEADER)
        for path in paths:
            with Path(path).open(encoding="utf-8",newline="") as source:
                reader=csv.reader(source); next(reader)
                writer.writerows(reader)


def run_job(job,output,allow_synthetic_cpu=False):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError("Use a new output directory; do not overwrite prior inference")
    if not output.is_relative_to(ROOT) and not allow_synthetic_cpu:
        raise ValueError("Output must stay inside Go Competition")
    if not Path(job["cache"]).resolve().is_relative_to(ROOT) and not allow_synthetic_cpu:
        raise ValueError("Feature cache must stay inside Go Competition")
    bundle=load_bundle(job["bundle"],allow_synthetic_cpu)
    games={}; pools=[load_pool(spec,games,bundle) for spec in job["pools"]]
    if len({p.name for p in pools})!=len(pools) or not pools: raise ValueError("Repeated or missing pools")
    qids=[q["question_id"] for p in pools for q in p.questions]
    if len(qids)!=len(set(qids)): raise ValueError("Question IDs must be unique across pools")
    method=job.get("method","color")
    if method not in ("color","original"): raise ValueError("Invalid selected method")
    output.mkdir(parents=True)
    state={"status":"started","started_utc":utc(),"no_submission":True,"development_only":job.get("development_only",False)}
    write_json(output/"status.json",state)
    write_json(output/"job.json",job)
    write_json(output/"input_manifest.json",{"bundle_manifest_sha256":bundle["manifest_sha256"],"pools":[p.source for p in pools],"code_hashes":bundle["manifest"]["code_hashes"]})
    start=time.perf_counter()
    try:
        state["status"]="parsing_sgf"; write_json(output/"status.json",state)
        features,keys,cache_stats=extract_features(games,bundle,job["cache"],job.get("workers",4),output)
        state["status"]="encoding"; write_json(output/"status.json",state)
        vectors,evidence=encode_features(features,bundle,job.get("device","cuda"))
        del features
        write_json(output/"no_training_evidence.json",evidence)
        positions={k:i for i,k in enumerate(keys)}; fallback_total=0
        for pool in pools:
            directory=output/pool.name; directory.mkdir()
            pairs=[component_scores(v,pool,games,positions) for v in vectors]
            original=rrf(pairs[0][0],pairs[1][0])
            color,selected,fallback=choose_scores(pairs[0][1],pairs[1][1],original,method)
            fallback_total+=int(fallback.sum())
            for name,scores in (("original_standardized",pairs[0][0]),("original_metric",pairs[1][0]),("original_rrf",original),("color_standardized",pairs[0][1]),("color_metric",pairs[1][1]),("color_rrf",color)):
                np.save(directory/f"{name}_scores.npy",scores,allow_pickle=False)
            np.save(directory/"color_row_fallback.npy",fallback,allow_pickle=False)
            for variant,scores in (("color",color),("original",original),("selected",selected)):
                write_top5(directory/f"top5_{variant}.csv",pool,scores)
            write_json(directory/"mapping.json",{"candidate_ids":pool.candidate_ids,"question_ids":[q["question_id"] for q in pool.questions],"query_num_games":[q["num_games"] for q in pool.questions],"candidate_order":"first appearance in candidate CSV","selected_method":method,"numeric_color_fallback_question_ids":[pool.questions[i]["question_id"] for i in np.flatnonzero(fallback)]})
            for component,(_,_,metadata) in zip(("statistical","learned"),pairs):
                np.savez(directory/f"{component}_color_metadata.npz",**metadata)
            print(json.dumps({"stage":"pool_scored","pool":pool.name,"questions":len(pool.questions)}),flush=True)
        checks={}
        for variant in ("color","original","selected"):
            target=output/f"submission_{variant}.csv"
            combine_files([output/p.name/f"top5_{variant}.csv" for p in pools],target)
            checks[variant]=check_output(target,pools)
        for p in pools:
            if sha(p.source["candidates_csv"])!=p.source["candidates_sha256"] or sha(p.source["queries_csv"])!=p.source["queries_sha256"]:
                raise ValueError("Input CSV changed during inference")
        if sha(bundle["path"]/"manifest.json")!=bundle["manifest_sha256"]:
            raise ValueError("Bundle manifest changed during inference")
        for rel,digest in bundle["manifest"]["files"].items():
            if sha(bundle["path"]/rel)!=digest: raise ValueError("Bundle payload changed during inference")
        for rel,digest in bundle["manifest"]["code_hashes"].items():
            if sha(ROOT/rel)!=digest: raise ValueError("Inference code changed during execution")
        write_json(output/"format_check.json",checks)
        result={"status":"complete","elapsed_seconds":time.perf_counter()-start,"pool_count":len(pools),"questions":len(qids),"unique_raw_color_games":len(games),"cache":cache_stats,"numeric_color_row_fallbacks":fallback_total,"selected_method":method,"retired_scope_guard_enabled":True,"no_submission":True}
        write_json(output/"summary.json",result)
        state.update(status="complete",completed_utc=utc()); write_json(output/"status.json",state)
        write_json(output/"output_sha256.json",{str(p.relative_to(output)).replace("\\","/"):sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
        return result
    except BaseException as error:
        state.update(status="failed",failed_utc=utc(),error=f"{type(error).__name__}: {error}",traceback=traceback.format_exc(),no_silent_game_drop=True)
        write_json(output/"status.json",state)
        raise
