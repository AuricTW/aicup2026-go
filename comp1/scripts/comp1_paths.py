"""Machine-specific settings for comp1.

Values are read from comp1/configs/paths.yaml (copy paths.example.yaml and edit it); an environment variable
AICUP_<KEY> (upper-case key) overrides the file. Relative paths are resolved against the comp1/ folder.
Only the scripts that start external processes or read files outside comp1/ need these settings; every other
script locates its inputs relative to comp1/ (data/, runs/, submissions/).
"""
import os
from pathlib import Path

COMP1 = Path(__file__).resolve().parents[1]


def _load():
    for name in ('paths.yaml', 'paths.example.yaml'):
        f = COMP1 / 'configs' / name
        if f.exists():
            out = {}
            for line in f.read_text(encoding='utf-8').splitlines():
                line = line.split(' #', 1)[0].strip()
                if line and not line.startswith('#') and ':' in line:
                    k, v = line.split(':', 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
            return out
    return {}


_CFG = _load()


def get(key, default=None):
    """Raw string setting (environment variable AICUP_<KEY> first, then the config file, then default)."""
    return os.environ.get('AICUP_' + key.upper()) or _CFG.get(key) or default


def path(key, default=None):
    """Path setting; relative values are taken relative to comp1/."""
    v = get(key, default)
    if v is None:
        return None
    p = Path(v)
    return p if p.is_absolute() else COMP1 / p
