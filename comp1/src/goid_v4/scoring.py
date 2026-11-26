"""Independent-query, fixed-order FP64 scoring; no GT, optimizer or file loaders."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .artifacts import ContractError
from .contracts import identifier


def finite_matrix(x, ndim=2):
    x = np.asarray(x)
    if x.ndim != ndim or x.dtype.kind not in 'fiu' or not np.isfinite(x).all():
        raise ContractError("Expected finite numeric matrix")
    x=x.astype(np.float64,copy=False)
    if not np.isfinite(x).all():raise ContractError('Value exceeds FP64 numeric contract')
    return x


def stable_ranks(scores):
    scores = finite_matrix(scores, 1)
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def rrf(stat_scores, mlp_scores):
    if np.shape(stat_scores) != np.shape(mlp_scores):
        raise ContractError("RRF shape mismatch")
    return .5 / (60 + stable_ranks(stat_scores)) + .5 / (60 + stable_ranks(mlp_scores))


def fixed_fusion(baseline_rrf, new_scores):
    b = finite_matrix(baseline_rrf, 1).astype(np.float64, copy=False)
    if b.shape != np.shape(new_scores):
        raise ContractError("Fusion shape mismatch")
    return .75 * b + .25 / (60 + stable_ranks(new_scores))


def top_five(scores, candidate_order, *, expected_candidates=400):
    if len(candidate_order) != expected_candidates or len(set(candidate_order)) != expected_candidates:
        raise ContractError("Candidate order incomplete or repeated")
    for p in candidate_order:
        identifier(p)
    scores = finite_matrix(scores, 1)
    if scores.shape != (expected_candidates,) or expected_candidates < 5:
        raise ContractError("Wrong score shape")
    return tuple(candidate_order[i] for i in np.argsort(-scores, kind="stable")[:5])


def _unit(x):
    norm = np.linalg.norm(x)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ContractError("Degenerate global prototype")
    return x / norm


@dataclass(frozen=True)
class GameSet:
    keys: tuple[str, ...]
    colors: tuple[str, ...]
    vectors: np.ndarray

    def canonical(self):
        x = finite_matrix(self.vectors).astype(np.float64, copy=True)
        n = len(self.keys)
        if n == 0 or len(set(self.keys)) != n or x.shape[0] != n or len(self.colors) != n:
            raise ContractError("Invalid set cardinality")
        if any(c not in ("B", "W") for c in self.colors):
            raise ContractError("Invalid set color")
        for key in self.keys:
            identifier(key)
        order = sorted(range(n), key=self.keys.__getitem__)
        x = np.ascontiguousarray(x[order]); x.setflags(write=False)
        return GameSet(tuple(self.keys[i] for i in order), tuple(self.colors[i] for i in order), x)


class PrototypeBank:
    def __init__(self, candidate_order, support_sets):
        self.order = tuple(candidate_order)
        if len(set(self.order)) != len(self.order) or set(support_sets) != set(self.order) or not self.order:
            raise ContractError("Bank identity mismatch")
        globals_, colors, missing = [], [], []
        dim = None
        self.support_keys = set()
        for p in self.order:
            s = support_sets[p].canonical()
            if self.support_keys & set(s.keys):
                raise ContractError("Repeated game key across candidates")
            self.support_keys.update(s.keys)
            if dim is None: dim = s.vectors.shape[1]
            if s.vectors.shape[1] != dim:
                raise ContractError("Mixed embedding dimensions")
            global_ = _unit(s.vectors.mean(axis=0))
            pair, flags = [], []
            for color in ("B", "W"):
                mask = np.asarray(s.colors) == color
                v = s.vectors[mask].mean(axis=0) if mask.any() else global_
                fallback = not mask.any() or np.linalg.norm(v) <= 1e-12
                pair.append(global_ if fallback else _unit(v)); flags.append(fallback)
            globals_.append(global_); colors.append(pair); missing.append(flags)
        self.global_ = np.ascontiguousarray(globals_, dtype=np.float64)
        self.colors = np.asarray(colors, dtype=np.float64)
        self.missing = np.asarray(missing)
        for x in (self.global_, self.colors, self.missing): x.setflags(write=False)

    def score_one(self, query: GameSet, *, color=False):
        q = query.canonical()
        if set(q.keys) & self.support_keys or q.vectors.shape[1] != self.global_.shape[1]:
            raise ContractError("Query/support overlap or dimension mismatch")
        if not color:
            return self.global_ @ _unit(q.vectors.mean(axis=0))
        scores = np.zeros(len(self.order), dtype=np.float64)
        for i, c in enumerate(("B", "W")):
            mask = np.asarray(q.colors) == c
            if mask.any():
                projected = np.ascontiguousarray(self.colors[:, i, :]) @ _unit(q.vectors[mask].mean(axis=0))
                scores += (int(mask.sum()) / len(q.keys)) * projected
        return scores

    def score_many(self, queries, *, color=False):
        # A fixed one-query path makes batching/subsetting mathematically and numerically explicit.
        return {qid: self.score_one(q, color=color) for qid, q in queries.items()}


class LegacyRRFReference:
    """FP64 synthetic reference, NOT a replacement/parity claim for v3 FP32 stacks."""
    def __init__(self, candidate_order, stat_supports, mlp_supports):
        self.stat = PrototypeBank(candidate_order, stat_supports)
        self.mlp = PrototypeBank(candidate_order, mlp_supports)
        for p in candidate_order:
            a,b=stat_supports[p].canonical(),mlp_supports[p].canonical()
            if a.keys!=b.keys or a.colors!=b.colors:
                raise ContractError("Legacy branches must use the same games/colors")

    def score_one(self, stat_query, mlp_query):
        a,b=stat_query.canonical(),mlp_query.canonical()
        if a.keys!=b.keys or a.colors!=b.colors:raise ContractError("Query branches misaligned")
        original=rrf(self.stat.score_one(a),self.mlp.score_one(b))
        reason=None
        try: color=rrf(self.stat.score_one(a,color=True),self.mlp.score_one(b,color=True))
        except ContractError:
            color=original.copy();reason="numeric_color_query_to_original"
        return {"original":original,"color":color,"numeric_row_fallback":reason,
                "candidate_color_fallbacks":int(self.stat.missing.sum()+self.mlp.missing.sum())}
