"""Run SAM egg segmentation over a set of images and write raw masks, overlays, a manifest.

Outputs under --out:
  masks/<image_id>.png      union of selected instances, uint8 {0,255}, original image size
  instances/<image_id>.png  uint8 label image (0 background, 1..N selected instances)
  overlays/<image_id>.jpg   image with selected instances in green and contours
  manifest.jsonl            one JSON record per image
Raw outputs are never edited by hand; reviewed masks live in a separate directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.data.inventory import inventory
from src.segmentation.masks import save_mask_png
from src.segmentation.sam_segmenter import SamSegmenter, SelectionConfig


def overlay(img: np.ndarray, inst_labels: np.ndarray) -> np.ndarray:
    out = img.copy()
    green = np.zeros_like(img)
    green[:] = (0, 255, 0)
    fg = inst_labels > 0
    out[fg] = (0.55 * img[fg] + 0.45 * green[fg]).astype(np.uint8)
    for k in range(1, int(inst_labels.max()) + 1):
        cs, _ = cv2.findContours((inst_labels == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, (0, 0, 255), 2)
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_raw")
    ap.add_argument("--checkpoint", default="models/sam_vit_b_01ec64.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", nargs="*", help="image_ids to run (default: all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--points-per-side", type=int, default=32)
    a = ap.parse_args()

    inv = inventory(Path(a.data))
    inv = inv[inv.readable]
    if a.ids:
        missing = set(a.ids) - set(inv.image_id)
        if missing:
            raise KeyError(f"unknown image ids: {sorted(missing)}")
        inv = inv[inv.image_id.isin(a.ids)]
    out = Path(a.out)
    ckpt = Path(a.checkpoint)
    cfg = SelectionConfig()
    import torch

    seg = SamSegmenter(ckpt, device=a.device, points_per_side=a.points_per_side, cfg=cfg)
    ckpt_sha = sha256_of(ckpt)
    records, t_all = [], time.time()
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for r in inv.itertuples():
        t0 = time.time()
        rec = {"image_id": r.image_id, "source_path": f"{a.data}/{r.path}", "source_sha256": r.sha256,
               "original_split": r.original_split}
        try:
            img = cv2.imdecode(np.fromfile(str(Path(a.data) / r.path), np.uint8), cv2.IMREAD_COLOR)
            mask, inst, n_cand = seg.segment(img)
            labels = np.zeros(img.shape[:2], np.uint8)
            for k, i in enumerate(inst, 1):
                labels[(i["mask"] > 0) & (labels == 0)] = k
            save_mask_png(out / "masks" / f"{r.image_id}.png", mask)
            (out / "instances").mkdir(parents=True, exist_ok=True)
            cv2.imencode(".png", labels)[1].tofile(str(out / "instances" / f"{r.image_id}.png"))
            (out / "overlays").mkdir(parents=True, exist_ok=True)
            cv2.imencode(".jpg", overlay(img, labels))[1].tofile(str(out / "overlays" / f"{r.image_id}.jpg"))
            rec.update(
                status="ok", mask_path=f"masks/{r.image_id}.png", height=int(img.shape[0]), width=int(img.shape[1]),
                n_candidates=n_cand, detection_count=len(inst), fg_fraction=float(mask.mean()),
                instances=[{"predicted_iou": round(i["predicted_iou"], 4), "stability": round(i["stability_score"], 4),
                            "area_frac": round(i["area_frac"], 4), "ellipse_iou": round(i["features"]["ellipse_iou"], 4),
                            "solidity": round(i["features"]["solidity"], 4), "aspect": round(i["features"]["aspect"], 3),
                            "touches_border": i["features"]["touches_border"]} for i in inst],
            )
        except Exception as e:  # record, never hide
            rec.update(status="error", error=f"{type(e).__name__}: {e}")
        rec["runtime_s"] = round(time.time() - t0, 3)
        records.append(rec)
        print(rec["image_id"], rec["status"], rec.get("detection_count"), rec["runtime_s"], flush=True)
    meta = {"model": "sam_vit_b", "checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha, "device": a.device,
            "points_per_side": a.points_per_side, "pred_iou_thresh": 0.80, "stability_score_thresh": 0.85,
            "selection": cfg.as_dict(), "torch": torch.__version__, "total_runtime_s": round(time.time() - t_all, 2),
            "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if a.device == "cuda" else None}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps({**rec, **{"run": meta}}) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
