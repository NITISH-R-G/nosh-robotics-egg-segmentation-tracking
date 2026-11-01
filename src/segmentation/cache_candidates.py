"""One-time SAM inference: cache ALL raw automatic-mask candidates, independent of any selection rule.

Layout under --out (format version 1):
  masks/<image_id>.npz   masks: uint8 (N, ceil(H*W/8)) np.packbits of row-major flattened bool masks,
                         shape: (H, W). Unpack with unpack_masks(). Candidate k <-> masks[k].
  candidates.jsonl       one JSON object per candidate (features below)
  run.json               model, checkpoint sha256, all generator settings, runtime, GPU memory
No candidate is dropped after generation. The generator itself applies pred_iou_thresh,
stability_score_thresh and its own box-NMS (box_nms_thresh=0.7); those are recorded in run.json.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.inventory import inventory
from src.segmentation.run_sam import sha256_of
from src.segmentation.sam_segmenter import SamSegmenter, shape_features

FORMAT_VERSION = 1


_IMG_RE = re.compile(r'^\{"image_id": "([^"]+)"')


def read_candidates(cache: Path, image_ids: set[str]) -> pd.DataFrame:
    """Read only the manifest lines of the requested images. Other lines (e.g. evaluation images) are skipped on the
    raw text before any JSON parsing, so they are never loaded. Sorted by (image_id, index) for determinism."""
    rows = []
    with open(Path(cache) / "candidates.jsonl") as f:
        for line in f:
            m = _IMG_RE.match(line)
            if m is None:
                raise ValueError(f"unrecognised candidates.jsonl line: {line[:60]!r}")
            if m.group(1) in image_ids:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["image_id", "index"])
    return pd.DataFrame(rows).sort_values(["image_id", "index"], kind="stable").reset_index(drop=True)


def pack_masks(masks: list[np.ndarray]) -> tuple[np.ndarray, tuple[int, int]]:
    shape = masks[0].shape
    return np.stack([np.packbits(np.asarray(m, bool).ravel()) for m in masks]), shape


def unpack_masks(npz_path: Path) -> np.ndarray:
    """Return bool array (N, H, W) from a cached npz."""
    with np.load(npz_path) as z:
        h, w = (int(v) for v in z["shape"])
        packed = z["masks"]
    return np.stack([np.unpackbits(r)[: h * w].reshape(h, w).astype(bool) for r in packed])


def candidate_record(image_id: str, k: int, npz_rel: str, mask: np.ndarray, c: dict) -> dict:
    m = np.ascontiguousarray(mask, dtype=np.uint8)
    f = shape_features(m)
    ys, xs = np.nonzero(m)
    return {
        "image_id": image_id, "candidate_id": f"{image_id}#{k}", "index": k, "mask_ref": npz_rel,
        "bbox_xywh": [float(v) for v in c["bbox_xywh"]], "area": int(f["area"]),
        "centroid_xy": [float(xs.mean()), float(ys.mean())] if len(xs) else None,
        "touches_border": f["touches_border"], "predicted_iou": c["predicted_iou"],
        "stability_score": c["stability_score"], "solidity": f["solidity"],
        "aspect": f["aspect"] if np.isfinite(f["aspect"]) else None, "ellipse_iou": f["ellipse_iou"],
        "point_coords_xy": [float(v) for v in c["point_coords"]], "crop_box": [int(v) for v in c["crop_box"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_raw")
    ap.add_argument("--checkpoint", default="models/sam_vit_b_01ec64.pth")
    ap.add_argument("--out", default="outputs/sam_candidates")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--pred-iou-thresh", type=float, default=0.5)
    ap.add_argument("--stability-thresh", type=float, default=0.5)
    ap.add_argument("--resume", action="store_true", help="skip images whose npz already exists")
    a = ap.parse_args()
    import torch

    inv = inventory(Path(a.data))
    inv = inv[inv.readable].sort_values("image_id")
    out = Path(a.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    seg = SamSegmenter(Path(a.checkpoint), device=a.device, points_per_side=a.points_per_side,
                       pred_iou_thresh=a.pred_iou_thresh, stability_score_thresh=a.stability_thresh)
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_all = time.time()
    per_image = {}
    mode = "a" if a.resume else "w"
    with open(out / "candidates.jsonl", mode) as fj:
        for r in inv.itertuples():
            npz = out / "masks" / f"{r.image_id}.npz"
            if a.resume and npz.exists():
                continue
            t0 = time.time()
            img = cv2.imdecode(np.fromfile(str(Path(a.data) / r.path), np.uint8), cv2.IMREAD_COLOR)
            cands = seg.candidates(img)
            if cands:
                packed, shape = pack_masks([c["mask"] for c in cands])
            else:
                packed, shape = np.zeros((0, 0), np.uint8), img.shape[:2]
            np.savez_compressed(npz, masks=packed, shape=np.array(shape))
            for k, c in enumerate(cands):
                fj.write(json.dumps(candidate_record(r.image_id, k, f"masks/{r.image_id}.npz", c["mask"], c)) + "\n")
            fj.flush()
            per_image[r.image_id] = {"n": len(cands), "s": round(time.time() - t0, 2)}
            print(r.image_id, len(cands), per_image[r.image_id]["s"], flush=True)
    run = {"format_version": FORMAT_VERSION, "model": "sam_vit_b", "checkpoint": a.checkpoint,
           "checkpoint_sha256": sha256_of(Path(a.checkpoint)), "device": a.device, "torch": torch.__version__,
           "points_per_side": a.points_per_side, "pred_iou_thresh": a.pred_iou_thresh,
           "stability_score_thresh": a.stability_thresh, "box_nms_thresh": 0.7, "crop_n_layers": 0,
           "min_mask_region_area": 0, "output_mode": "binary_mask", "seed": 0,
           "total_runtime_s": round(time.time() - t_all, 1), "per_image": per_image,
           "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if a.device == "cuda" else None}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    print("done", run["total_runtime_s"], "s")


if __name__ == "__main__":
    main()
