"""Offline egg-candidate selection over cached raw SAM candidates (feature-based, no model needed).

Three kinds of mask exist and must never be mixed up:
  1. raw SAM candidates (outputs/sam_candidates, immutable),
  2. automatically selected egg masks (this module),
  3. human-reviewed reference masks (separate directory, with a correction manifest).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectorConfig:
    name: str
    min_area_frac: float = 0.004
    max_area_frac: float = 0.70
    min_solidity: float = 0.90
    max_aspect: float = 2.0
    min_ellipse_iou: float = 0.85
    min_pred_iou: float = 0.80
    min_stability: float = 0.85
    # Optional stricter/looser rule for candidates that touch the image border (clipped eggs fit an ellipse worse).
    border_min_ellipse_iou: float | None = None
    border_min_area_frac: float | None = None
    nms_iou: float = 0.5
    nms_contain: float = 0.8
    fallback: str = "none"  # none | review | relaxed_top1
    relaxed_min_ellipse_iou: float = 0.80
    relaxed_min_pred_iou: float = 0.75
    # A candidate is kept if it passes this config's gate OR the complete gate of any alt (each alt is a full
    # SelectorConfig used only for its thresholds; nested alts/fallbacks are ignored).
    alts: tuple = ()
    # Used only when fallback == "alt_gate": if the primary gate selects nothing in an image, try these gates
    # (with the same NMS) instead of picking a single best candidate.
    fallback_alts: tuple = ()
    # Drop selected instances whose area is below this fraction of the largest selected instance in the image
    # (only applies when more than one instance is selected).
    min_rel_area: float | None = None

    def validate(self) -> None:
        if not 0 <= self.min_area_frac < self.max_area_frac <= 1:
            raise ValueError("need 0 <= min_area_frac < max_area_frac <= 1")
        if self.fallback not in {"none", "review", "relaxed_top1", "alt_gate"}:
            raise ValueError(f"unknown fallback: {self.fallback}")
        if self.max_aspect < 1:
            raise ValueError("max_aspect must be >= 1")
        if self.min_rel_area is not None and not 0 < self.min_rel_area < 1:
            raise ValueError("min_rel_area must be in (0, 1)")
        for a in self.alts + self.fallback_alts:
            a.validate()
        if self.fallback == "alt_gate" and not self.fallback_alts:
            raise ValueError("fallback='alt_gate' needs fallback_alts")

    def as_dict(self) -> dict:
        return asdict(self)


def _pass(row: pd.Series, area_frac: float, cfg: SelectorConfig) -> bool:
    return _pass_one(row, area_frac, cfg) or any(_pass_one(row, area_frac, a) for a in cfg.alts)


def _pass_one(row: pd.Series, area_frac: float, cfg: SelectorConfig) -> bool:
    if not (cfg.min_area_frac <= area_frac <= cfg.max_area_frac):
        return False
    aspect = row["aspect"]
    if pd.isna(aspect) or aspect > cfg.max_aspect:
        return False
    if row["solidity"] < cfg.min_solidity or row["predicted_iou"] < cfg.min_pred_iou:
        return False
    if row["stability_score"] < cfg.min_stability:
        return False
    e_min = cfg.min_ellipse_iou
    if row["touches_border"]:
        if cfg.border_min_ellipse_iou is not None:
            e_min = cfg.border_min_ellipse_iou
        if cfg.border_min_area_frac is not None and area_frac < cfg.border_min_area_frac:
            return False
    return row["ellipse_iou"] >= e_min


def _nms(order: list[int], masks: Callable[[int], np.ndarray], cfg: SelectorConfig) -> list[int]:
    kept: list[int] = []
    for i in order:
        mi = masks(i)
        dup = False
        for k in kept:
            mk = masks(k)
            inter = float((mi & mk).sum())
            union = float((mi | mk).sum())
            smaller = float(min(mi.sum(), mk.sum()))
            if (union and inter / union > cfg.nms_iou) or (smaller and inter / smaller > cfg.nms_contain):
                dup = True
                break
        if not dup:
            kept.append(i)
    return kept


def select_from_features(dfi: pd.DataFrame, masks: Callable[[int], np.ndarray], image_area: int,
                         cfg: SelectorConfig) -> tuple[list[int], str]:
    """dfi: candidate rows of ONE image (columns as in candidates.jsonl; index = candidate 'index').
    masks(index) -> bool (H, W). Returns (selected candidate indices ranked by score, status).
    status: ok | none_selected | review_no_candidate | fallback_top1.
    """
    cfg.validate()
    if len(dfi) == 0:
        return [], ("none_selected" if cfg.fallback == "none" else "review_no_candidate")
    score =dfi["ellipse_iou"] * dfi["predicted_iou"]
    ok = [i for i, r in dfi.iterrows() if _pass(r, r["area"] / image_area, cfg)]
    ok.sort(key=lambda i: -score[i])
    sel = _nms(ok, masks, cfg)
    if cfg.min_rel_area is not None and len(sel) > 1:
        biggest = max(dfi.loc[i, "area"] for i in sel)
        sel = [i for i in sel if dfi.loc[i, "area"] >= cfg.min_rel_area * biggest]
    if sel:
        return sel, "ok"
    if cfg.fallback == "alt_gate":
        ok2 = [i for i, r in dfi.iterrows() if any(_pass_one(r, r["area"] / image_area, a) for a in cfg.fallback_alts)]
        ok2.sort(key=lambda i: -score[i])
        sel2 = _nms(ok2, masks, cfg)
        if sel2:
            return sel2, "alt_gate_fallback"
        return [], "review_no_candidate"
    if cfg.fallback == "review":
        return [], "review_no_candidate"
    if cfg.fallback == "relaxed_top1":
        relaxed = SelectorConfig(
            name=cfg.name + "_relaxed", min_area_frac=cfg.min_area_frac, max_area_frac=cfg.max_area_frac,
            min_solidity=0.90, max_aspect=cfg.max_aspect, min_ellipse_iou=cfg.relaxed_min_ellipse_iou,
            min_pred_iou=cfg.relaxed_min_pred_iou, min_stability=0.80)
        cand = [i for i, r in dfi.iterrows() if _pass(r, r["area"] / image_area, relaxed)]
        if cand:
            return [max(cand, key=lambda i: score[i])], "fallback_top1"
        return [], "review_no_candidate"
    return [], "none_selected"
