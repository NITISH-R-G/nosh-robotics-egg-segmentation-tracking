"""Per-frame egg detector for the video task (Task 4).

Deliberately separate from the frozen image-set segmenters: the video has one
fixed look (white eggs on a dark table, a hand, a checked shirt), so this
detector is a plain fixed-threshold + shape-filter pipeline whose parameters
are set on development frames only (see experiments/video_tracking_protocol.md).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoDetectorConfig:
    work_width: int = 960          # frames are downscaled to this width before detection
    min_gray: int = 200            # egg pixels are near-white
    max_sat: int = 60              # and unsaturated (HSV S, 0-255)
    open_k: int = 5
    close_k: int = 7
    min_area_frac: float = 0.002   # of the working frame
    max_area_frac: float = 0.05
    min_solidity: float = 0.90
    max_aspect: float = 1.7
    min_fill_ellipse: float = 0.85  # component area / area of its fitted ellipse

    def sha(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def detect(frame_bgr: np.ndarray, cfg: VideoDetectorConfig) -> list[dict]:
    """Return detections in ORIGINAL frame coordinates, sorted by x then y (deterministic)."""
    h0, w0 = frame_bgr.shape[:2]
    scale = cfg.work_width / w0
    img = cv2.resize(frame_bgr, (cfg.work_width, int(round(h0 * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    m = ((gray >= cfg.min_gray) & (sat <= cfg.max_sat)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_k,) * 2))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_k,) * 2))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    total = m.size
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (cfg.min_area_frac * total <= area <= cfg.max_area_frac * total):
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if len(c) < 5:
            continue
        (_, _), (ma, mi), _ = cv2.fitEllipse(c)
        aspect = max(ma, mi) / max(min(ma, mi), 1e-6)
        fill = area / (np.pi * ma * mi / 4.0)
        if solidity < cfg.min_solidity or aspect > cfg.max_aspect or fill < cfg.min_fill_ellipse:
            continue
        x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        out.append({
            "bbox": [x / scale, y / scale, (x + w) / scale, (y + h) / scale],
            "centroid": [float(cent[i][0] / scale), float(cent[i][1] / scale)],
            "area": area / scale ** 2, "solidity": float(solidity), "aspect": float(aspect), "fill": float(fill),
            "_mask": comp, "_scale": scale,
        })
    out.sort(key=lambda d: (round(d["centroid"][0]), round(d["centroid"][1])))
    return out
