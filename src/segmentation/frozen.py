"""Frozen selector configuration: stored with a content hash so it cannot drift silently."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.segmentation.selector import SelectorConfig


def canonical_sha256(cfg_dict: dict) -> str:
    return hashlib.sha256(json.dumps(cfg_dict, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def config_from_dict(d: dict) -> SelectorConfig:
    d = dict(d)
    d["alts"] = tuple(config_from_dict(a) for a in d.get("alts", ()))
    d["fallback_alts"] = tuple(config_from_dict(a) for a in d.get("fallback_alts", ()))
    return SelectorConfig(**d)


def write_frozen(path: Path, cfg: SelectorConfig, meta: dict) -> dict:
    d = cfg.as_dict()
    doc = {"selector": d, "selector_sha256": canonical_sha256(d), **meta}
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True))
    return doc


def load_frozen(path: Path) -> tuple[SelectorConfig, dict]:
    doc = json.loads(Path(path).read_text())
    if canonical_sha256(doc["selector"]) != doc["selector_sha256"]:
        raise ValueError(f"frozen selector config {path} was modified after freezing (hash mismatch)")
    cfg = config_from_dict(doc["selector"])
    cfg.validate()
    return cfg, doc
