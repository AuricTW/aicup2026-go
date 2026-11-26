"""Fixed GPU/CPU batch CSV runtime; frozen v1 ingestion and RRF are reused."""
from pathlib import Path
import time
import traceback

import numpy as np

from goid import inference as base, fixed_batch as fixed, row_projection as projection


def load_bundle(path, allow_synthetic_cpu=False):
    path=Path(path).resolve(); manifest=base.read_json(path/"manifest.json")
    if manifest.get("schema")!="goid-fixed-batch-execution-v3": raise ValueError("Expected fixed-batch v3 execution bundle")
    if base.read_json(path/"manifest.sha256.json")["manifest.json"]!=base.sha(path/"manifest.json"):
        raise ValueError("V3 manifest checksum failed")
    if manifest["encoding_policy"]!=fixed.POLICY: raise ValueError("Encoding policy differs")
    if manifest["projection_policy"]!=projection.POLICY: raise ValueError("Projection policy differs")
    for rel,h in manifest["code_hashes"].items():
        if base.sha(base.ROOT/rel)!=h: raise ValueError(f"Pinned v3 source changed: {rel}")
    for rel,h in manifest["files"].items():
        target=(path/rel).resolve()
        if not target.is_relative_to(path) or base.sha(target)!=h: raise ValueError("V3 payload checksum failed")
    bundle=base.load_bundle(path/manifest["base_bundle"],allow_synthetic_cpu)
    if bundle["manifest_sha256"]!=manifest["base_manifest_sha256"]: raise ValueError("V3 base bundle changed")
    bundle.update(execution_path=path,execution_manifest=manifest,execution_manifest_sha256=base.sha(path/"manifest.json"))
    return bundle


def verify_bundle(bundle):
    for rel,h in bundle["manifest"]["files"].items():
        if base.sha(bundle["path"]/rel)!=h: raise ValueError("Base bundle payload changed during inference")
    if base.sha(bundle["path"]/"manifest.json")!=bundle["manifest_sha256"]: raise ValueError("Base manifest changed")
    if base.sha(bundle["execution_path"]/"manifest.json")!=bundle["execution_manifest_sha256"]: raise ValueError("V3 manifest changed")
    for manifest in (bundle["manifest"],bundle["execution_manifest"]):
        for rel,h in manifest["code_hashes"].items():
            if base.sha(base.ROOT/rel)!=h: raise ValueError("Pinned runtime source changed")
    for rel,h in bundle["execution_manifest"]["files"].items():
        if base.sha(bundle["execution_path"]/rel)!=h: raise ValueError("V3 policy changed")


def run_job(job,output,allow_synthetic_cpu=False):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError("Preserve prior inference; use new output")
    if not allow_synthetic_cpu and (not output.is_relative_to(base.ROOT) or not Path(job["cache"]).resolve().is_relative_to(base.ROOT)):
        raise ValueError("Output/cache must stay inside Go Competition")
    if any(k in job for k in ("batch_size","batch_rows","pad_tail","query_batch_rows")): raise ValueError("V3 batch policy is pinned, not a job option")
    bundle=load_bundle(job["bundle"],allow_synthetic_cpu)
    games={}; pools=[base.load_pool(spec,games,bundle) for spec in job["pools"]]
    qids=[q["question_id"] for p in pools for q in p.questions]
    if not pools or len({p.name for p in pools})!=len(pools) or len(qids)!=len(set(qids)): raise ValueError("Missing/repeated pool or query IDs")
    method=job.get("method","color")
    if method not in ("color","original"): raise ValueError("Invalid method")
    output.mkdir(parents=True)
    state={"status":"started","started_utc":base.utc(),"no_submission":True,"development_only":job.get("development_only",False),"runtime_version":"v3"}
    base.write_json(output/"status.json",state); base.write_json(output/"job.json",job)
    base.write_json(output/"input_manifest.json",{"base_bundle_manifest_sha256":bundle["manifest_sha256"],"execution_manifest_sha256":bundle["execution_manifest_sha256"],"encoding_policy":fixed.POLICY,"projection_policy":projection.POLICY,"pools":[p.source for p in pools],"base_code_hashes":bundle["manifest"]["code_hashes"],"v3_code_hashes":bundle["execution_manifest"]["code_hashes"]})
    start=time.perf_counter()
    try:
        state["status"]="parsing_sgf"; base.write_json(output/"status.json",state)
        features,keys,cache_stats=base.extract_features(games,bundle,job["cache"],job.get("workers",4),output)
        if keys!=sorted(keys) or len(keys)!=len(set(keys)): raise ValueError("Feature content order violated")
        state["status"]="encoding"; base.write_json(output/"status.json",state)
        vectors,evidence=fixed.encode_features(features,bundle,job.get("device","cuda")); del features
        evidence["projection_policy"]=projection.POLICY
        base.write_json(output/"no_training_evidence.json",evidence)
        positions={k:i for i,k in enumerate(keys)}; fallback_total=0
        for pool in pools:
            directory=output/pool.name; directory.mkdir()
            pairs=[projection.component_scores(v,pool,games,positions) for v in vectors]
            original=base.rrf(pairs[0][0],pairs[1][0])
            color,selected,fallback=base.choose_scores(pairs[0][1],pairs[1][1],original,method)
            fallback_total+=int(fallback.sum())
            for name,scores in (("original_standardized",pairs[0][0]),("original_metric",pairs[1][0]),("original_rrf",original),("color_standardized",pairs[0][1]),("color_metric",pairs[1][1]),("color_rrf",color)):
                np.save(directory/f"{name}_scores.npy",scores,allow_pickle=False)
            np.save(directory/"color_row_fallback.npy",fallback,allow_pickle=False)
            for variant,scores in (("color",color),("original",original),("selected",selected)): base.write_top5(directory/f"top5_{variant}.csv",pool,scores)
            base.write_json(directory/"mapping.json",{"candidate_ids":pool.candidate_ids,"question_ids":[q["question_id"] for q in pool.questions],"query_num_games":[q["num_games"] for q in pool.questions],"candidate_order":"first appearance in candidate CSV","selected_method":method,"numeric_color_fallback_question_ids":[pool.questions[i]["question_id"] for i in np.flatnonzero(fallback)]})
            for component,(_,_,metadata) in zip(("statistical","learned"),pairs): np.savez(directory/f"{component}_color_metadata.npz",**metadata)
            print(base.json.dumps({"stage":"pool_scored","pool":pool.name,"questions":len(pool.questions)}),flush=True)
        checks={}
        for variant in ("color","original","selected"):
            target=output/f"submission_{variant}.csv"
            base.combine_files([output/p.name/f"top5_{variant}.csv" for p in pools],target)
            checks[variant]=base.check_output(target,pools)
        for pool in pools:
            for field in ("candidates","queries"):
                if base.sha(pool.source[f"{field}_csv"])!=pool.source[f"{field}_sha256"]: raise ValueError("Input CSV changed")
        verify_bundle(bundle)
        base.write_json(output/"format_check.json",checks)
        result={"status":"complete","runtime_version":"v3","encoding_policy":fixed.POLICY["name"],"projection_policy":projection.POLICY["name"],"elapsed_seconds":time.perf_counter()-start,"pool_count":len(pools),"questions":len(qids),"unique_raw_color_games":len(games),"cache":cache_stats,"numeric_color_row_fallbacks":fallback_total,"selected_method":method,"retired_scope_guard_enabled":True,"no_submission":True}
        base.write_json(output/"summary.json",result)
        state.update(status="complete",completed_utc=base.utc()); base.write_json(output/"status.json",state)
        base.write_json(output/"output_sha256.json",{p.relative_to(output).as_posix():base.sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
        return result
    except BaseException as error:
        state.update(status="failed",failed_utc=base.utc(),error=f"{type(error).__name__}: {error}",traceback=traceback.format_exc(),no_silent_game_drop=True)
        base.write_json(output/"status.json",state); raise
