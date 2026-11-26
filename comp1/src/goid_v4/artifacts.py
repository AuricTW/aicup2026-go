"""Immutable artifacts and explicitly bounded, checksum-pinned metadata reads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ContractError(ValueError):
    pass


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       allow_nan=False, indent=2) + "\n").encode("utf-8")


def object_hash(value) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def bytes_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unique_pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ContractError(f"Duplicate JSON key: {k}")
        result[k] = v
    return result


def strict_json(raw: bytes):
    def reject(value):
        raise ContractError(f"Nonfinite JSON constant: {value}")
    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_pairs,
                       parse_constant=reject)
    try:
        json_bytes(value)  # Also rejects exponent overflow such as 1e999.
    except ValueError as exc:
        raise ContractError("Nonfinite JSON number") from exc
    return value


class ArtifactStore:
    """Exclusive creation within one explicit new output root; no overwrite."""
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative: str) -> Path:
        p = Path(relative)
        if p.is_absolute() or ".." in p.parts or not p.parts:
            raise ContractError("Artifact path must be relative and cannot ascend")
        target = (self.root / p).resolve()
        if not target.is_relative_to(self.root) or target == self.root:
            raise ContractError("Artifact escapes output root")
        return target

    def write(self, relative: str, raw: bytes):
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create also rejects a race that creates the target first.
        with target.open("xb") as f:
            f.write(raw)
        return {"path": relative, "bytes": len(raw), "sha256": bytes_hash(raw)}

    def write_json(self, relative: str, value):
        return self.write(relative, json_bytes(value))


class PinnedMetadataReader:
    """No globbing, fallback search or arbitrary-path read. No model/data loader."""
    def __init__(self, root: Path, assets: dict[str, dict]):
        self.root = Path(root).resolve()
        self.assets = dict(assets)
        self.access_log = []

    def read(self, asset: str):
        if asset not in self.assets:
            raise ContractError("Input is not in explicit metadata allowlist")
        spec = self.assets[asset]
        if spec.get("kind") != "safe_metadata_json":
            raise ContractError("Only predeclared safe metadata JSON is accepted")
        rel = Path(spec["path"])
        path = (self.root / rel).resolve()
        if rel.is_absolute() or ".." in rel.parts or not path.is_relative_to(self.root):
            raise ContractError("Metadata path escapes declared root")
        if path.suffix.lower() != ".json" or path.stat().st_size > spec["max_bytes"]:
            raise ContractError("Metadata suffix or byte budget mismatch")
        raw = path.read_bytes()
        digest = bytes_hash(raw)
        if digest != spec["sha256"]:
            raise ContractError("Metadata checksum mismatch; do not repin silently")
        value = strict_json(raw)
        self.access_log.append({"asset": asset, "path": spec["path"],
                                "bytes": len(raw), "sha256": digest})
        return value


def validate_cache_manifest(manifest, expected, store: ArtifactStore):
    semantic_keys = ("scope_hash", "encoder_hash", "teacher_hash", "profile_hash",
                     "precision_hash", "parser_hash", "representation_hash")
    if set(expected) != set(semantic_keys):
        raise ContractError("Every cache semantic key must be declared")
    for k in semantic_keys:
        if manifest.get(k) != expected[k]:
            raise ContractError(f"Cache invalidated by {k}")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ContractError("Cache needs explicit shards")
    names = set()
    for row in shards:
        name = row["path"]
        if name in names:
            raise ContractError("Repeated cache shard")
        names.add(name)
        path = store.path(name)
        if path.stat().st_size != row["bytes"] or bytes_hash(path.read_bytes()) != row["sha256"]:
            raise ContractError("Cache shard changed")
        for k in ("unique_games", "unique_groups", "valid_positions"):
            if type(row[k]) is not int or row[k] < 0:
                raise ContractError("Invalid cache counts")
        if not isinstance(row["shape"], list) or any(type(x) is not int or x < 0 for x in row["shape"]):
            raise ContractError("Invalid cache shape")
    return True
