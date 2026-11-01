"""Detector D1 for the tracking-improvement variants: the unchanged strict detector plus 'partial' white fragments.

`detect.py` is not modified. Complete detections are exactly the strict detector's output (same code path); partial
detections are extra white components that failed the whole-egg shape filters but are large enough to be part of an egg.
Rules are pre-registered in experiments/tracking_improvement_audit.md section 5.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .detect import VideoDetectorConfig, detect


@dataclass(frozen=True)
class PartialConfig:
    min_area_frac: float = 0.003    # of the 960 px working frame (about 20% of a whole egg)
    max_area_frac: float = 0.05
    min_solidity: float = 0.65
    max_aspect: float = 3.5
    max_partial: int = 3
    enabled: bool = True

    def sha(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def detect_with_partials(frame_bgr: np.ndarray, base: VideoDetectorConfig, part: PartialConfig) -> list[dict]:
    """Return detections in original coordinates with a 'kind' key: 'complete' (identical to detect()) or 'partial'."""
    dets = [dict(d, kind="complete") for d in detect(frame_bgr, base)]
    if not part.enabled:
        return dets
    h0, w0 = frame_bgr.shape[:2]
    scale = base.work_width / w0
    img = cv2.resize(frame_bgr, (base.work_width, int(round(h0 * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    m = ((gray >= base.min_gray) & (sat <= base.max_sat)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base.open_k,) * 2))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base.close_k,) * 2))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    total = m.size
    complete_boxes = [d["bbox"] for d in dets]
    cands = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (part.min_area_frac * total <= area <= part.max_area_frac * total):
            continue
        x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        bbox = [x / scale, y / scale, (x + w) / scale, (y + h) / scale]
        cx, cy = cent[i][0] / scale, cent[i][1] / scale
        if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in complete_boxes):
            continue                      # already part of a complete detection
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull if hull > 0 else 0.0
        if len(c) < 5:
            continue
        (_, _), (ma, mi), _ = cv2.fitEllipse(c)
        aspect = max(ma, mi) / max(min(ma, mi), 1e-6)
        if solidity < part.min_solidity or aspect > part.max_aspect:
            continue
        cands.append({"bbox": bbox, "centroid": [float(cx), float(cy)], "area": area / scale ** 2, "solidity": float(solidity),
                      "aspect": float(aspect), "fill": float("nan"), "kind": "partial"})
    cands.sort(key=lambda d: -d["area"])
    out = dets + cands[: part.max_partial]
    out.sort(key=lambda d: (d["kind"] != "complete", round(d["centroid"][0]), round(d["centroid"][1])))
    return out
