"""Interpretable appearance descriptors for candidate masks, and group-aware separation statistics.

Hypotheses (written BEFORE measuring; direction = which side is more egg-like):
  H_L       mean Lab lightness inside the mask is HIGH        (white/cream shells are bright)
  H_chroma  mean Lab chroma inside the mask is LOW            (shells are nearly unsaturated; cloth, tile, bowls, foil are not)
  H_dL      inside minus ring lightness is HIGH               (an egg is usually brighter than what surrounds it; fails for backlit eggs)
  H_texture mean gradient magnitude inside the eroded mask is LOW  (smooth shell vs woven cloth, tile pattern, rubble, keys)
  H_edge    mean gradient magnitude on the mask boundary is HIGH   (a real object edge; weak for white on white)
No colour threshold is hand-set: thresholds are only ever chosen from training scenes in `loso_threshold`.
These descriptors were invented after seeing the reviewed failures, so every result is EXPLORATORY; leave-one-scene-out
evaluation is the only guard against overfitting to a scene.
"""
from __future__ import annotations

import numpy as np

ORIENT = {"L_in": +1, "chroma_in": -1, "dL_ring": +1, "texture_in": -1, "edge_strength": +1}
GEOMETRY = {"ellipse_iou": +1, "solidity": +1, "predicted_iou": +1, "stability_score": +1}


def _lab(img_bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def descriptors(img_bgr: np.ndarray, mask: np.ndarray, ring_px: int = 12, gap_px: int = 3) -> dict:
    """Appearance descriptors of one candidate mask on its image (same (H, W) frame)."""
    import cv2

    m = np.asarray(mask, bool)
    if img_bgr.shape[:2] != m.shape:
        raise ValueError("image and mask shapes differ")
    if not m.any():
        raise ValueError("empty mask")
    lab = _lab(img_bgr)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    mu8 = m.astype(np.uint8)

    def se(r):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    inner = cv2.erode(mu8, se(4), borderValue=0).astype(bool)
    inner = inner if inner.sum() >= 30 else m
    ring = cv2.dilate(mu8, se(ring_px)).astype(bool) & ~cv2.dilate(mu8, se(gap_px)).astype(bool)
    if ring.sum() < 30:  # mask fills the frame: fall back to everything outside the mask
        ring = ~m
    gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    band = cv2.dilate(mu8, se(2)).astype(bool) & ~cv2.erode(mu8, se(2), borderValue=0).astype(bool)
    return {
        "L_in": float(L[m].mean()), "chroma_in": float(chroma[m].mean()),
        "dL_ring": float(L[m].mean() - L[ring].mean()) if ring.any() else float("nan"),
        "texture_in": float(grad[inner].mean()), "edge_strength": float(grad[band].mean()),
    }


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) with ties counted half (Mann-Whitney). NaN if either class is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_threshold(score: np.ndarray, y: np.ndarray) -> float:
    """Threshold on a higher-is-more-egg score maximising balanced accuracy (midpoints between distinct values)."""
    s = np.unique(score)
    if len(s) == 1:
        return float(s[0])
    cands = np.concatenate([[s[0] - 1], (s[:-1] + s[1:]) / 2, [s[-1] + 1]])
    pos, neg = y == 1, y == 0
    best, bt = -1.0, cands[0]
    for t in cands:
        bal = 0.5 * ((score[pos] >= t).mean() + (score[neg] < t).mean())
        if bal > best:
            best, bt = bal, t
    return float(bt)


def loso_threshold(score: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Leave-one-group-out predictions: the threshold for each group is chosen only from the OTHER groups."""
    pred = np.zeros(len(score), int)
    for g in np.unique(groups):
        te = groups == g
        t = best_threshold(score[~te], y[~te])
        pred[te] = (score[te] >= t).astype(int)
    return pred


def cluster_bootstrap_auc(score, y, groups, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval of pooled AUC when whole groups (scenes) are resampled; honest about few, correlated scenes."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        a = auc(score[idx][y[idx] == 1], score[idx][y[idx] == 0])
        if not np.isnan(a):
            vals.append(a)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))
