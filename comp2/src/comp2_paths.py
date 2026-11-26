"""Machine-specific settings for comp2.

Values are read from comp2/configs/paths.yaml (copy paths.example.yaml and edit it); an environment variable
AICUP2_<KEY> (upper-case key) overrides the file. Relative paths are resolved against the comp2/ folder.
"""
import os
from pathlib import Path

COMP2 = Path(__file__).resolve().parents[1]
THIRD_PARTY = COMP2.parent / "third_party"


def _load():
    for name in ("paths.yaml", "paths.example.yaml"):
        f = COMP2 / "configs" / name
        if f.exists():
            out = {}
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.split(" #", 1)[0].strip()
                if line and not line.startswith("#") and ":" in line:
                    k, v = line.split(":", 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
            return out
    return {}


_CFG = _load()


def get(key, default=None):
    """Raw string setting (environment variable AICUP2_<KEY> first, then the config file, then default)."""
    return os.environ.get("AICUP2_" + key.upper()) or _CFG.get(key) or default


def path(key, default=None):
    """Path setting; relative values are taken relative to comp2/."""
    v = get(key, default)
    if v is None:
        return None
    p = Path(v)
    return p if p.is_absolute() else COMP2 / p


def katago_engine(env):
    """(KataGo executable, environment) for the analysis engine; katago_lib_dir is prepended to LD_LIBRARY_PATH if set."""
    lib = get("katago_lib_dir")
    if lib:
        env["LD_LIBRARY_PATH"] = lib + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return str(path("katago_exe", "katago")), env


ROOT = str(path("root", "."))                 # holds data/raw/... (official files) and data/interim/index.csv
WORK = path("work", "work")                   # intermediate files (datasets, model outputs, evidence files)
D64_OOF = str(path("d64_oof", "prev_system/OLD_FUSED.npz"))
D64_Q = str(path("d64_queries", "prev_system/queries_dense.json"))
D64_TEST = str(path("d64_test", "prev_system/go_rank_D64_profile5_parent5_logG_cal6_RESEARCH_20261107_v1_posteriors.npz"))
