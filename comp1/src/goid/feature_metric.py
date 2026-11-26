"""Small learned metric over frozen statistical features, not a board CNN."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from goid.episodes import EpisodePlan, Question


class FeatureMetric(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, features):
        return F.normalize(self.network(features), dim=-1, eps=1e-8)


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """All other games of the same player are positives; self is excluded."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if embeddings.ndim != 2 or labels.ndim != 1 or len(embeddings) != len(labels):
        raise ValueError("Expected [batch,dimension] embeddings and [batch] labels")
    diagonal = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    counts = positives.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Each anchor must have another game from its player")
    # Explicitly disable the caller's autocast: .float() alone does not prevent
    # autocast matmul from returning FP16 (where -1e9 would overflow).
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        vectors = embeddings.float()
        logits = (vectors @ vectors.T) / temperature
        logits = logits.masked_fill(diagonal, -1e9)
        log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return -(log_probabilities.masked_fill(~positives, 0).sum(dim=1) / counts).mean()


class PlayerBalancedSampler:
    def __init__(self, row_to_player, players_per_batch=16, games_per_player=4, seed=20261025):
        if players_per_batch < 2 or games_per_player < 2:
            raise ValueError("SupCon needs at least two players and two games per player")
        groups = {}
        for row, player in row_to_player.items():
            groups.setdefault(player, []).append(row)
        self.all_players = tuple(sorted(groups))
        self.labels = {player: index for index, player in enumerate(self.all_players)}
        self.eligible = tuple(player for player in self.all_players if len(groups[player]) >= games_per_player)
        if len(self.eligible) < players_per_batch:
            raise ValueError(f"Need {players_per_batch} fit players with {games_per_player} unique games; found {len(self.eligible)}")
        self.rows = {player: np.asarray(sorted(groups[player]), dtype=np.int64) for player in self.eligible}
        self.p = players_per_batch
        self.k = games_per_player
        self.rng = np.random.default_rng(seed)

    def sample(self):
        selected = self.rng.choice(len(self.eligible), size=self.p, replace=False)
        row_ids, labels = [], []
        for index in selected:
            player = self.eligible[index]
            row_ids.extend(self.rng.choice(self.rows[player], size=self.k, replace=False))
            labels.extend([self.labels[player]] * self.k)
        return np.asarray(row_ids, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def read_fixed_plan(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    plan = EpisodePlan(
        tuple(value["candidate_ids"]),
        {player: tuple(rows) for player, rows in value["support_row_ids"].items()},
        tuple(Question(q["question_id"], q["player_id"], q["query_games"], tuple(q["query_row_ids"])) for q in value["questions"]),
        value["report"],
    )
    if len(set(plan.candidate_ids)) != len(plan.candidate_ids) or len(plan.candidate_ids) < 5:
        raise ValueError("Invalid fixed candidate pool")
    if set(plan.support_row_ids) != set(plan.candidate_ids):
        raise ValueError("Support keys differ from candidate pool")
    support_rows = [row for rows in plan.support_row_ids.values() for row in rows]
    if len(set(support_rows)) != len(support_rows):
        raise ValueError("Fixed support pool repeats a row")
    all_queries = set()
    seen_questions = set()
    for question in plan.questions:
        if question.player_id not in plan.candidate_ids or question.question_id in seen_questions:
            raise ValueError("Invalid or repeated fixed question")
        seen_questions.add(question.question_id)
        if len(question.query_row_ids) != question.query_games or len(set(question.query_row_ids)) != question.query_games:
            raise ValueError("Wrong or repeated query games")
        all_queries.update(question.query_row_ids)
    if set(support_rows) & all_queries:
        raise ValueError("Support/query overlap in fixed episodes")
    return plan


def checked_fit_index(audit_path, fit_ids, player_splits, plan):
    """Read fit/validation rows, checking full-game or supplied family groups."""
    fit_ids = np.asarray(fit_ids, dtype=np.int64)
    if len(set(fit_ids.tolist())) != len(fit_ids) or np.any(fit_ids < 0):
        raise ValueError("Invalid or repeated fit row IDs")
    validation_owner = {}
    for player, rows in plan.support_row_ids.items():
        for row in rows:
            validation_owner[row] = player
    for question in plan.questions:
        for row in question.query_row_ids:
            previous = validation_owner.setdefault(row, question.player_id)
            if previous != question.player_id:
                raise ValueError("A query row belongs to multiple candidate players")
    if set(fit_ids.tolist()) & set(validation_owner):
        raise ValueError("Fit rows overlap validation episodes")
    connection = sqlite3.connect(Path(audit_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(games)")}
        grouping_expression = "COALESCE(split_group_hash,canonical_hash)" if "split_group_hash" in columns else "canonical_hash"

        def fetch(rows):
            rows = list(rows)
            for start in range(0, len(rows), 900):
                batch = rows[start:start + 900]
                placeholders = ",".join("?" for _ in batch)
                yield from connection.execute(f"SELECT row_id,player_id,{grouping_expression},parse_ok FROM games WHERE row_id IN ({placeholders})", batch)

        validation_canonicals = set()
        validation_seen = set()
        for row, player, canonical, parse_ok in fetch(validation_owner):
            if parse_ok != 1 or player != validation_owner[row] or player_splits.get(player) != "validation":
                raise ValueError(f"Fixed validation row violates parse/player split: {row}")
            if not canonical or canonical in validation_canonicals:
                raise ValueError("Fixed validation contains missing/repeated effective game group")
            validation_canonicals.add(canonical)
            validation_seen.add(row)
        if validation_seen != set(validation_owner):
            raise ValueError("Validation rows missing from audit")
        result, fit_canonicals = {}, set()
        for row, player, canonical, parse_ok in fetch(fit_ids.tolist()):
            if parse_ok != 1 or player_splits.get(player) != "fit":
                raise ValueError(f"Fixed fit row violates parse/player split: {row}")
            if not canonical or canonical in fit_canonicals or canonical in validation_canonicals:
                raise ValueError("Fit effective game group repeated or overlaps validation")
            fit_canonicals.add(canonical)
            result[row] = player
        if len(result) != len(fit_ids):
            raise ValueError("Fit rows missing from audit")
        return result
    finally:
        connection.close()


class FrozenFeaturePreprocessor:
    """Reuse statistical run's exact fit statistics; never fit on validation."""
    def __init__(self, statistical_run, mode="standardized", clip=10.0):
        root = Path(statistical_run)
        schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8-sig"))
        groups = schema if isinstance(schema, list) else schema.get("groups", schema.get("feature_groups"))
        with np.load(root / "fit_transform.npz", allow_pickle=False) as arrays:
            self.indices = arrays["selected_indices"].copy()
            self.mean = arrays["mean"].copy()
            self.std = arrays["std"].copy()
        if len(self.indices) != len(self.mean) or len(self.mean) != len(self.std) or np.any(self.std <= 0):
            raise ValueError("Invalid frozen fit transform")
        positions = {int(column): index for index, column in enumerate(self.indices)}
        self.histograms = []
        for group in groups:
            if group["kind"] == "histogram":
                selected = [positions[column] for column in range(group["start"], group["stop"]) if column in positions]
                if selected:
                    self.histograms.append(np.asarray(selected, dtype=np.int64))
        self.mode, self.clip = mode, clip

    def __call__(self, features):
        value = np.array(features[:, self.indices], dtype=np.float32, copy=True)
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("Nonfinite or negative statistical features")
        for columns in self.histograms:
            histogram = value[:, columns]
            value[:, columns] = histogram / np.maximum(histogram.sum(axis=1, keepdims=True), 1e-12)
        if self.mode == "standardized":
            value = np.clip((value - self.mean) / self.std, -self.clip, self.clip)
        elif self.mode == "sqrt":
            value = np.sqrt(value)
        elif self.mode != "raw":
            raise ValueError(f"Unknown preprocessing mode {self.mode}")
        value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
        return np.ascontiguousarray(value)


def encode_plan(model, features, preprocess, plan, device, batch_size=1024):
    row_ids = sorted({row for rows in plan.support_row_ids.values() for row in rows} | {row for q in plan.questions for row in q.query_row_ids})
    positions = {row: index for index, row in enumerate(row_ids)}
    embeddings = []
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(row_ids), batch_size):
            matrix = preprocess(features[row_ids[start:start + batch_size]])
            embeddings.append(model(torch.from_numpy(matrix).to(device)).float().cpu().numpy())
    model.train(was_training)
    embeddings = np.concatenate(embeddings)

    def profile(rows):
        vector = embeddings[[positions[row] for row in rows]].mean(axis=0)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    supports = np.stack([profile(plan.support_row_ids[player]) for player in plan.candidate_ids])
    queries = np.stack([profile(q.query_row_ids) for q in plan.questions])
    return queries @ supports.T
