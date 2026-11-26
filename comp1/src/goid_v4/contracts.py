"""Fail-closed scopes, episode isolation, and text CSV row schemas (no file I/O)."""
from __future__ import annotations

from dataclasses import dataclass
import re
from collections import defaultdict

from .artifacts import ContractError

EXPOSURES = ("representation_fit", "normalizer_fit", "policy_pretrain",
             "checkpoint_selection", "score_seen", "error_seen", "protocol_selection")
RULE_FLAGS = ("external_data", "external_weights", "candidate_finetuning", "query_tta",
              "cross_query_statistics", "pseudolabel", "joint_assignment")
HASH_FIELDS = ("raw_token", "move_token", "canonical_token", "effective_group_token")


def identifier(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError("Expected nonempty unpadded identifier")
    return value


def integer(value, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ContractError("Integer out of declared range")
    return value


def validate_rules(registry):
    if set(registry) != set(RULE_FLAGS) or any(v is not False for v in registry.values()):
        raise ContractError("Engineering mainline requires each external rule flag explicitly false")


def validate_exposure(flags):
    if set(flags) != set(EXPOSURES) | {"unknown"}:
        raise ContractError("Incomplete exposure ledger")
    if any(v is not None and type(v) is not bool for v in flags.values()):
        raise ContractError("Exposure values are true/false/null")
    if type(flags["unknown"]) is not bool:
        raise ContractError("unknown must be explicit")
    if any(flags[k] is None for k in EXPOSURES) and not flags["unknown"]:
        raise ContractError("Null exposure cannot be called known")
    return not flags["unknown"] and all(flags[k] is False for k in EXPOSURES)


@dataclass(frozen=True)
class AllowedGame:
    source_row_key: str
    opaque_player_key: str
    target_color: str
    raw_token: str
    move_token: str
    canonical_token: str
    effective_group_token: str
    role: str
    rank_train_only: str | None
    exposure: dict
    scope_hash: str

    @classmethod
    def parse(cls, row):
        if set(row) != set(cls.__dataclass_fields__):
            raise ContractError("allowed_games fields must match exact schema")
        result = cls(**row)
        identifier(result.source_row_key)
        identifier(result.opaque_player_key)
        if result.target_color not in ("B", "W") or result.role not in ("fit", "development", "outer"):
            raise ContractError("Invalid color or data role")
        for key in HASH_FIELDS + ("scope_hash",):
            if not isinstance(row[key], str) or not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ContractError("Expected SHA256 token")
        if result.role != "fit" and result.rank_train_only is not None:
            raise ContractError("Rank labels cannot enter deployment/evaluation scope")
        if result.rank_train_only is not None:
            identifier(result.rank_train_only)
        validate_exposure(result.exposure)
        return result


def validate_scope(games):
    if not games:
        raise ContractError("Empty legal game scope")
    if any(not isinstance(g,AllowedGame) for g in games):
        raise ContractError('Expected parsed allowed-game records')
    index, players, owners = {}, {}, {}
    scope_hashes = {g.scope_hash for g in games}
    if len(scope_hashes) != 1:
        raise ContractError("Mixed scope versions")
    for g in games:
        if not isinstance(g, AllowedGame):
            raise ContractError("Expected parsed allowed-game records")
        AllowedGame.parse(dict(g.__dict__))
        if g.source_row_key in index:
            raise ContractError("Duplicate global row key")
        index[g.source_row_key] = g
        if g.opaque_player_key in players and players[g.opaque_player_key] != g.role:
            raise ContractError("Player crosses fit/development/outer")
        players[g.opaque_player_key] = g.role
        for k in HASH_FIELDS:
            token = (k, getattr(g, k))
            if token in owners and owners[token] != g.role:
                raise ContractError(f"Known {k} crosses roles")
            owners[token] = g.role
    return index


def validate_episode(index, candidate_order, supports, queries, *, support_count=100,
                     query_count=20, candidate_count=400):
    if len(candidate_order) != candidate_count or len(set(candidate_order)) != candidate_count:
        raise ContractError("Candidate count/order mismatch")
    if set(supports) != set(candidate_order) or set(queries) != set(candidate_order):
        raise ContractError("Every candidate needs support and query")
    tokens = {"support": set(), "query": set()}
    scopes = set()
    for role, mapping, count in (("support", supports, support_count), ("query", queries, query_count)):
        for player in candidate_order:
            keys = mapping[player]
            if len(keys) != count or len(set(keys)) != count:
                raise ContractError("Repeated or insufficient game rows")
            local_groups = set()
            for key in keys:
                if key not in index or index[key].opaque_player_key != player:
                    raise ContractError("Episode row outside legal owner scope")
                g = index[key]
                scopes.add(g.role)
                if g.effective_group_token in local_groups:
                    raise ContractError("Repeated effective group within set")
                local_groups.add(g.effective_group_token)
                tokens[role].update((k, getattr(g, k)) for k in HASH_FIELDS)
    if len(scopes) != 1 or tokens["support"] & tokens["query"]:
        raise ContractError("Episode roles/group overlap")
    return True


def capacity(games):
    """Distinct effective groups per legal player, never counts duplicate rows twice."""
    validate_scope(games)
    grouped = defaultdict(set)
    for g in games:
        grouped[g.opaque_player_key].add(g.effective_group_token)
    return {str(q): sum(len(v) >= 100 + q for v in grouped.values()) for q in (5, 10, 20)}


def candidate_rows(rows, *, candidates=400, supports=100):
    fields = {"game_id", "player_id", "sgf_content", "color"}
    result, keys = defaultdict(list), set()
    for ordinal, row in enumerate(rows):
        if set(row) != fields or row["color"] not in ("B", "W"):
            raise ContractError("Candidate schema/color mismatch")
        p, g, text = (identifier(row[k]) for k in ("player_id", "game_id", "sgf_content"))
        # game_id uniqueness is not elevated to an unconfirmed official rule.
        row_token=(p,g,text,row['color'])
        if row_token in keys:
            raise ContractError("Repeated identical candidate row")
        keys.add(row_token)
        result[p].append({'source_row_ordinal':ordinal,'game_id':g,'sgf_content':text,'color':row['color']})
    if len(result) != candidates or any(len(v) != supports for v in result.values()):
        raise ContractError("Incomplete candidate pool")
    return dict(result)


def query_rows(rows):
    fields = {"question_id", "num_games"} | {f"{k}_{i}" for k in ("sgf", "color") for i in range(1, 21)}
    result = {}
    for row in rows:
        if set(row) != fields:
            raise ContractError("Query schema mismatch")
        qid = identifier(row["question_id"])
        raw_n = row["num_games"]
        if not isinstance(raw_n, str) or not raw_n.isdecimal():
            raise ContractError("num_games must be decimal CSV text")
        n = integer(int(raw_n), 5, 20)
        if qid in result:
            raise ContractError("Repeated question_id")
        games = []
        for i in range(1, n + 1):
            if row[f"color_{i}"] not in ("B", "W"):
                raise ContractError("Query color mismatch")
            games.append((identifier(row[f"sgf_{i}"]), row[f"color_{i}"]))
        result[qid] = games  # Deliberately never parse undeclared slots.
    if not result:
        raise ContractError("Empty query source")
    return result
