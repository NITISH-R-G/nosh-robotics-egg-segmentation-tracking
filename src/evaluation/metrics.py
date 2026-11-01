"""Segmentation, instance and count metrics used identically for the classical method and for S1.

Pixel metrics are AGREEMENT with a named reference (pseudo-ground-truth derived from SAM), never ground-truth accuracy.
Empty masks are handled explicitly: nothing divides by zero silently.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def _bin(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {m.shape}")
    return m.astype(bool)


def pixel_metrics(ref: np.ndarray, pred: np.ndarray) -> dict:
    """tp/fp/fn/tn pixel counts plus IoU, Dice, precision, recall.

    ref empty: iou/dice/recall are NaN (undefined); precision is 0.0 if pred is non-empty (all FP) else NaN; `ref_empty` is True.
    ref non-empty and pred empty: IoU = Dice = recall = 0.0, precision NaN (nothing predicted). Shapes must match.
    """
    r, p = _bin(ref), _bin(pred)
    if r.shape != p.shape:
        raise ValueError(f"shape mismatch {r.shape} vs {p.shape}")
    tp, fp, fn = int((r & p).sum()), int((~r & p).sum()), int((r & ~p).sum())
    tn = int(r.size - tp - fp - fn)
    nan = float("nan")
    ref_empty, pred_empty = not r.any(), not p.any()
    union = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "ref_empty": ref_empty, "pred_empty": pred_empty,
            "iou": nan if ref_empty else tp / union,
            "dice": nan if ref_empty else 2 * tp / (2 * tp + fp + fn),
            "precision": nan if pred_empty else tp / (tp + fp),
            "recall": nan if ref_empty else tp / (tp + fn)}


def connected_instances(mask: np.ndarray, min_area: int = 1) -> list[np.ndarray]:
    """8-connected components of a binary mask as boolean masks, ordered by (top, left) of their bounding box."""
    m = _bin(mask).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = [(stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_LEFT], i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return [lab == i for _, _, i in sorted(comps)]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _bin(a), _bin(b)
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def match_instances(pred: list[np.ndarray], ref: list[np.ndarray], iou_thr: float = 0.5) -> dict:
    """Hungarian assignment on mask IoU; a pair counts as a match only if IoU >= iou_thr.

    Returns matches [(pred_idx, ref_idx, iou)], false-positive pred indices, missed ref indices and tp/fp/fn counts.
    A merged prediction covering two reference eggs can match at most one of them (the other is a miss)."""
    if not pred or not ref:
        return {"matches": [], "fp_idx": list(range(len(pred))), "fn_idx": list(range(len(ref))), "tp": 0, "fp": len(pred), "fn": len(ref)}
    iou = np.array([[mask_iou(p, r) for r in ref] for p in pred])
    rows, cols = linear_sum_assignment(-iou)
    matches = [(int(i), int(j), float(iou[i, j])) for i, j in zip(rows, cols) if iou[i, j] >= iou_thr]
    mp, mr = {m[0] for m in matches}, {m[1] for m in matches}
    fp_idx = [i for i in range(len(pred)) if i not in mp]
    fn_idx = [j for j in range(len(ref)) if j not in mr]
    return {"matches": matches, "fp_idx": fp_idx, "fn_idx": fn_idx, "tp": len(matches), "fp": len(fp_idx), "fn": len(fn_idx)}


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    return {"precision": p, "recall": r, "f1": f}


def count_metrics(pred_counts: np.ndarray, expected: np.ndarray) -> dict:
    """Exact accuracy, mean/median/max absolute error, over/under rates for image-level egg counts."""
    p, e = np.asarray(pred_counts, int), np.asarray(expected, int)
    if p.shape != e.shape or p.size == 0:
        raise ValueError("count arrays must be non-empty and the same length")
    d = p - e
    return {"n_images": int(p.size), "exact_rate": float((d == 0).mean()), "over_rate": float((d > 0).mean()),
            "under_rate": float((d < 0).mean()), "mean_abs_err": float(np.abs(d).mean()),
            "median_abs_err": float(np.median(np.abs(d))), "max_abs_err": int(np.abs(d).max())}
