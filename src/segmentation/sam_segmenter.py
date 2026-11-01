"""Task 1: egg segmentation with the official Segment Anything Model (SAM).

SAM is class-agnostic: it returns object-like masks, not "eggs". Egg selection here is a
documented, annotation-free heuristic (shape + SAM confidence), see SelectionConfig.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SelectionConfig:
    min_area_frac: float = 0.004
    max_area_frac: float = 0.70
    min_solidity: float = 0.90
    max_aspect: float = 2.0
    min_ellipse_iou: float = 0.85
    min_pred_iou: float = 0.80
    min_stability: float = 0.85
    nms_iou: float = 0.5
    nms_contain: float = 0.8

    def validate(self) -> None:
        if not 0 <= self.min_area_frac < self.max_area_frac <= 1:
            raise ValueError("need 0 <= min_area_frac < max_area_frac <= 1")
        for k in ("min_solidity", "min_ellipse_iou", "min_pred_iou", "min_stability", "nms_iou", "nms_contain"):
            if not 0 <= getattr(self, k) <= 1:
                raise ValueError(f"{k} must be in [0, 1]")
        if self.max_aspect < 1:
            raise ValueError("max_aspect must be >= 1")

    def as_dict(self) -> dict:
        return asdict(self)


def shape_features(mask: np.ndarray) -> dict:
    """Shape descriptors of the largest contour of a {0,1} mask (all in image units)."""
    m = np.ascontiguousarray(mask, dtype=np.uint8)  # OpenCV drawing calls reject F-ordered arrays
    area = float(m.sum())
    out = {"area": area, "solidity": 0.0, "aspect": float("inf"), "ellipse_iou": 0.0, "touches_border": False}
    if area == 0:
        return out
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    out["solidity"] = float(cv2.contourArea(c) / hull_area) if hull_area > 0 else 0.0
    ys, xs = np.nonzero(m)
    out["touches_border"] = bool(
        ys.min() == 0 or xs.min() == 0 or ys.max() == m.shape[0] - 1 or xs.max() == m.shape[1] - 1
    )
    if len(c) >= 5:
        (cx, cy), (w, h), ang = cv2.fitEllipse(c)
        out["aspect"] = float(max(w, h) / max(min(w, h), 1e-6))
        ell = np.zeros_like(m)
        cv2.ellipse(ell, ((cx, cy), (w, h), ang), 1, -1)
        inter = float((ell & m).sum())
        union = float((ell | m).sum())
        out["ellipse_iou"] = inter / union if union else 0.0
    return out


def select_candidates(cands: list[dict], image_shape: tuple[int, int], cfg: SelectionConfig) -> list[dict]:
    """Filter and de-duplicate candidates.

    Each candidate needs 'mask' (bool/{0,1}), 'predicted_iou', 'stability_score'. Returns kept
    candidates with 'features', 'area_frac' and a ranking 'score' added. Deterministic.
    """
    cfg.validate()
    h, w = image_shape
    kept = []
    for c in cands:
        m = np.ascontiguousarray(c["mask"], dtype=np.uint8)
        f = shape_features(m)
        frac = f["area"] / (h * w)
        ok = (
            cfg.min_area_frac <= frac <= cfg.max_area_frac
            and f["solidity"] >= cfg.min_solidity
            and f["aspect"] <= cfg.max_aspect
            and f["ellipse_iou"] >= cfg.min_ellipse_iou
            and c["predicted_iou"] >= cfg.min_pred_iou
            and c["stability_score"] >= cfg.min_stability
        )
        if ok:
            kept.append(
                {**c, "mask": m, "features": f, "area_frac": frac, "score": f["ellipse_iou"] * c["predicted_iou"]}
            )
    kept.sort(key=lambda c: -c["score"])
    final: list[dict] = []
    for c in kept:
        dup = False
        for k in final:
            inter = float((c["mask"] & k["mask"]).sum())
            iou = inter / float((c["mask"] | k["mask"]).sum())
            contain = inter / min(c["mask"].sum(), k["mask"].sum())
            if iou > cfg.nms_iou or contain > cfg.nms_contain:
                dup = True
                break
        if not dup:
            final.append(c)
    return final


def union_mask(instances: list[dict], image_shape: tuple[int, int]) -> np.ndarray:
    """Foreground = union of selected instances; all-zero (not an error) when none selected."""
    out = np.zeros(image_shape, np.uint8)
    for i in instances:
        out |= np.asarray(i["mask"]).astype(np.uint8)
    return out


class SamSegmenter:
    """Wraps segment_anything's SamAutomaticMaskGenerator with deterministic settings."""

    def __init__(self, checkpoint: Path, model_type: str = "vit_b", device: str = "cuda",
                 points_per_side: int = 32, cfg: SelectionConfig | None = None,
                 pred_iou_thresh: float = 0.80, stability_score_thresh: float = 0.85):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
        torch.manual_seed(0)
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device).eval()
        self.points_per_side, self.model_type, self.device = points_per_side, model_type, device
        self.pred_iou_thresh, self.stability_score_thresh = pred_iou_thresh, stability_score_thresh
        self.cfg = cfg or SelectionConfig()
        self.generator = SamAutomaticMaskGenerator(
            sam, points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=0, output_mode="binary_mask")

    def candidates(self, image_bgr: np.ndarray) -> list[dict]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return [{"mask": r["segmentation"], "predicted_iou": float(r["predicted_iou"]),
                 "stability_score": float(r["stability_score"]), "bbox_xywh": r["bbox"],
                 "point_coords": r["point_coords"][0], "crop_box": r["crop_box"]}
                for r in self.generator.generate(rgb)]

    def segment(self, image_bgr: np.ndarray) -> tuple[np.ndarray, list[dict], int]:
        """Returns (union mask {0,1}, selected instances, number of raw candidates)."""
        if image_bgr is None or image_bgr.ndim != 3:
            raise ValueError("expected a BGR image array")
        cands = self.candidates(image_bgr)
        inst = select_candidates(cands, image_bgr.shape[:2], self.cfg)
        return union_mask(inst, image_bgr.shape[:2]), inst, len(cands)
