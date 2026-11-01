"""Classical (non-learned) egg segmentation: foreground map -> threshold -> morphology -> components -> filters -> optional split.

Rationale per operation (eggs in this dataset are white or cream, mostly brighter and less saturated than their surroundings):
  foreground map F   one channel in which eggs are expected to be HIGH: Lab lightness L (default), HSV value V, whiteness (L minus chroma),
                     or inverted HSV saturation. A single channel keeps the method interpretable and the thresholds few.
  threshold          Otsu (global, no hand-set level) or Gaussian adaptive (local mean + offset) for uneven lighting; optional Gaussian blur
                     first to suppress texture before thresholding.
  morphology         opening removes specks and thin bridges, closing fills small gaps; holes inside a component are filled (highlights, specular spots).
  components         8-connected components are the instance candidates.
  filters            area range (drops specks and whole-frame regions), solidity and aspect ratio (egg-like convex compact shape), border rule
                     (a clipped egg is large), ellipse fit (shape prior). Each is OFF by default and switched on one at a time by the experiments.
  split              distance-transform watershed separates touching eggs, only when enabled.
Known and expected failure modes: white-on-white, reflective foil, pale tile and cloth (the egg is not the brightest object),
backlit eggs (darker than the background) and submerged eggs. The method returns what the pipeline finds; it never invents eggs.
Deterministic: no randomness, only OpenCV operations on the input array.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np

COLOURS = {"L", "V", "W", "S"}
THRESHOLDS = {"otsu", "adaptive"}


@dataclass(frozen=True)
class ClassicalConfig:
    colour: str = "L"
    blur_sigma: float = 0.0
    threshold: str = "otsu"
    adaptive_block: int = 151
    adaptive_c: float = -5.0
    open_k: int = 5
    close_k: int = 5
    min_area_frac: float = 0.005
    max_area_frac: float = 0.60
    min_solidity: float = 0.0       # 0 = off
    max_aspect: float = 0.0         # 0 = off
    border_min_area_frac: float = 0.0  # 0 = off: components touching the border must be at least this large
    split_watershed: bool = False
    split_h: float = 0.4
    min_ellipse_iou: float = 0.0    # 0 = off

    def validate(self) -> None:
        if self.colour not in COLOURS or self.threshold not in THRESHOLDS:
            raise ValueError(f"unknown colour/threshold: {self.colour}/{self.threshold}")
        if self.adaptive_block < 3 or self.adaptive_block % 2 == 0:
            raise ValueError("adaptive_block must be an odd integer >= 3")
        if self.open_k < 0 or self.close_k < 0 or self.blur_sigma < 0:
            raise ValueError("kernel sizes and sigma must be >= 0")
        if not 0 <= self.min_area_frac < self.max_area_frac <= 1:
            raise ValueError("need 0 <= min_area_frac < max_area_frac <= 1")
        for k in ("min_solidity", "border_min_area_frac", "min_ellipse_iou"):
            if not 0 <= getattr(self, k) <= 1:
                raise ValueError(f"{k} must be in [0, 1]")
        if self.max_aspect != 0 and self.max_aspect < 1:
            raise ValueError("max_aspect must be 0 (off) or >= 1")
        if not 0 < self.split_h < 1:
            raise ValueError("split_h must be in (0, 1)")

    def as_dict(self) -> dict:
        return asdict(self)

    def sha(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:12]


def foreground_map(img_bgr: np.ndarray, colour: str) -> np.ndarray:
    """uint8 map, higher = more egg-like."""
    if colour in ("L", "W"):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        L = lab[..., 0]
        if colour == "L":
            return L
        chroma = np.sqrt((lab[..., 1].astype(np.float32) - 128) ** 2 + (lab[..., 2].astype(np.float32) - 128) ** 2)
        return np.clip(L.astype(np.float32) - chroma, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return hsv[..., 2] if colour == "V" else (255 - hsv[..., 1])


def _threshold(F: np.ndarray, cfg: ClassicalConfig) -> tuple[np.ndarray, float]:
    if int(F.max()) == int(F.min()):  # uniform image: nothing to separate (Otsu would pick an arbitrary level)
        return np.zeros(F.shape, np.uint8), float("nan")
    if cfg.threshold == "otsu":
        t, m = cv2.threshold(F, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return (m > 0).astype(np.uint8), float(t)
    m = cv2.adaptiveThreshold(F, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, cfg.adaptive_block, cfg.adaptive_c)
    return (m > 0).astype(np.uint8), float("nan")


def _kernel(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _fill_holes(m: np.ndarray) -> np.ndarray:
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cs, -1, 1, thickness=-1)
    return out


def _split(mask: np.ndarray, h_frac: float) -> np.ndarray:
    """Distance-transform watershed. Seeds = the parts of each component whose distance is at least (1 - h_frac) x the component's
    largest distance ("peak height" h_frac x max): two blobs get separate seeds only if the neck between their peaks is lower than
    that level, i.e. the peaks rise at least h_frac x max above the neck. (skimage's h_maxima was rejected: it keeps BOTH of two equal
    peaks whatever h is.) Flooding runs on -distance restricted to the mask, the canonical formulation (OpenCV's watershed floods on
    colour gradients, not on this landscape). Returns an int label image (0 = background)."""
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    n, comp, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    seeds = np.zeros(mask.shape, bool)
    for i in range(1, n):
        cm = comp == i
        mx = float(dist[cm].max())
        if mx > 0:
            seeds |= cm & (dist >= (1.0 - h_frac) * mx)
    markers, _ = ndi.label(seeds, structure=np.ones((3, 3), int))
    return watershed(-dist, markers, mask=mask > 0).astype(np.int32)


def _shape_stats(cm: np.ndarray) -> dict:
    cs, _ = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    solidity = float(cv2.contourArea(c) / hull) if hull > 0 else 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    aspect = float(max(w, h) / max(min(w, h), 1e-6))
    ell = 0.0
    if len(c) >= 5:
        e = cv2.fitEllipse(c)
        em = np.zeros(cm.shape, np.uint8)
        cv2.ellipse(em, e, 1, -1)
        inter, union = float((em.astype(bool) & cm).sum()), float((em.astype(bool) | cm).sum())
        ell = inter / union if union else 0.0
    return {"solidity": solidity, "aspect": aspect, "ellipse_iou": ell}


def segment_classical(img_bgr: np.ndarray, cfg: ClassicalConfig) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Returns (union mask uint8 {0,1}, instance masks (bool, ordered by top/left), info)."""
    cfg.validate()
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.dtype != np.uint8:
        raise ValueError("expected a uint8 BGR image")
    img = np.ascontiguousarray(img_bgr)
    h, w = img.shape[:2]
    F = foreground_map(img, cfg.colour)
    if cfg.blur_sigma > 0:
        F = cv2.GaussianBlur(F, (0, 0), cfg.blur_sigma)
    m, thr = _threshold(F, cfg)
    if cfg.open_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel(cfg.open_k))
    if cfg.close_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel(cfg.close_k))
    m = _fill_holes(m)
    if cfg.split_watershed and m.any():
        lab = _split(m, cfg.split_h)
        labels = [lab == i for i in range(1, int(lab.max()) + 1)]
    else:
        n, comp = cv2.connectedComponents(m, connectivity=8)
        labels = [comp == i for i in range(1, n)]
    amin, amax = cfg.min_area_frac * h * w, cfg.max_area_frac * h * w
    kept, why = [], {"area": 0, "solidity": 0, "aspect": 0, "border": 0, "ellipse": 0}
    for cm in labels:
        a = int(cm.sum())
        if a == 0:
            continue
        if not amin <= a <= amax:
            why["area"] += 1
            continue
        s = _shape_stats(cm)
        ys, xs = np.nonzero(cm)
        border = ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1
        if cfg.min_solidity and s["solidity"] < cfg.min_solidity:
            why["solidity"] += 1
        elif cfg.max_aspect and s["aspect"] > cfg.max_aspect:
            why["aspect"] += 1
        elif cfg.border_min_area_frac and border and a < cfg.border_min_area_frac * h * w:
            why["border"] += 1
        elif cfg.min_ellipse_iou and s["ellipse_iou"] < cfg.min_ellipse_iou:
            why["ellipse"] += 1
        else:
            kept.append((int(ys.min()), int(xs.min()), cm))
    inst = [cm for _, _, cm in sorted(kept, key=lambda t: (t[0], t[1]))]
    union = np.zeros((h, w), np.uint8)
    for cm in inst:
        union[cm] = 1
    return union, inst, {"threshold": thr, "components": len(labels), "kept": len(inst), "rejected": why}
