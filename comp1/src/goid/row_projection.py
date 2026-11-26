"""One query per GEMV avoids query-batch size and within-batch row position."""
import numpy as np

POLICY = {
    "name": "one_query_gemv_projection_v3",
    "query_rows_per_call": 1,
    "dtype": "float32",
    "candidate_layout": "C contiguous copy per global/black/white prototype matrix",
    "query_layout": "C contiguous independent copy for every query",
    "operation": "candidate_matrix @ query_vector, same shape for every query in a pool",
    "candidate_count": "actual count, unchanged within the same pool",
    "padding": "No query padding; real prototypes and actual query color weights only",
    "scope": "Fixed local NumPy/BLAS stack; no cross-hardware bitwise guarantee",
}


def dot(queries,candidates):
    if queries.ndim!=2 or candidates.ndim!=2 or queries.shape[1]!=candidates.shape[1]: raise ValueError("Incompatible prototype matrices")
    if queries.dtype!=np.float32 or candidates.dtype!=np.float32: raise ValueError("Expected FP32 prototypes")
    support=np.array(candidates,dtype=np.float32,order="C",copy=True)
    result=np.empty((len(queries),len(candidates)),np.float32)
    for i,query in enumerate(queries):
        real=np.array(query,dtype=np.float32,order="C",copy=True)
        result[i]=support @ real
    return result


def component_scores(vectors,pool,games,positions):
    # Reuse the same prototype implementation without mutating its module.
    # This local closure receives the new dot operator explicitly.
    return _component_scores(vectors,pool,games,positions,dot)


def _component_scores(vectors,pool,games,positions,project):
    from goid.inference import normalize_rows
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
    original=project(np.stack(qglobal),support[:,0])
    color=weights[:,:1]*project(qcolors[:,0],support[:,1])+weights[:,1:]*project(qcolors[:,1],support[:,2])
    return original,color,{"support_color_counts":counts,"support_fallback":fallback,"query_color_counts":qcounts,"query_color_weights":weights}
