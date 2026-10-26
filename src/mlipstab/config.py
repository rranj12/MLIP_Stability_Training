"""YAML configs with `includes:` resolved relative to the including file."""

from pathlib import Path

import yaml


def _merge(base, override):
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    merged = {}
    for inc in cfg.pop("includes", []):
        merged = _merge(merged, load_config(path.parent / inc))
    return _merge(merged, cfg)
