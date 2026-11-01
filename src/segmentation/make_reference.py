"""Apply the FROZEN selector to every image and write the raw automatic reference masks (Task 1 output).

Raw automatic masks are never edited. Reviewed corrections live in a separate directory built by
build_reviewed.py from a manifest of manual selections. Applying the frozen selector to eval images is
part of producing the reference; eval outputs must not be inspected for tuning.

Outputs under --out: masks/<id>.png ({0,255}, original size), instances/<id>.png (labels),
manifest.jsonl (one record per image: split, status, selected candidate ids, hashes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.segmentation.cache_candidates import unpack_masks
from src.segmentation.frozen import load_frozen
from src.segmentation.masks import save_mask_png
from src.segmentation.selector import select_from_features


def mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(mask, bool).ravel()).tobytes()).hexdigest()


def make_reference(cache: Path, split_csv: Path, frozen_json: Path, out: Path, audit_json: Path | None = None) -> pd.DataFrame:
    cfg, doc = load_frozen(frozen_json)
    if audit_json is not None:
        audit = json.loads(Path(audit_json).read_text())
        if not audit.get("ok"):
            raise RuntimeError("cache audit did not pass")
        for name, h in audit["stats"]["hashes"].items():
            if hashlib.sha256((cache / name).read_bytes()).hexdigest() != h:
                raise RuntimeError(f"stale audit: {name} changed")
    split = pd.read_csv(split_csv)
    cands = pd.read_json(cache / "candidates.jsonl", lines=True)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in split.sort_values("image_id").itertuples():
        dfi = cands[cands.image_id == r.image_id].set_index("index")
        npz = cache / "masks" / f"{r.image_id}.npz"
        allm = unpack_masks(npz) if len(dfi) else None
        img = cv2.imdecode(np.fromfile(str(Path("data_raw") / r.path), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        sel, status = select_from_features(dfi, (lambda i: allm[i]) if allm is not None else (lambda i: None), h * w, cfg)
        union = np.zeros((h, w), np.uint8)
        labels = np.zeros((h, w), np.uint8)
        for k, i in enumerate(sel, 1):
            union |= allm[i].astype(np.uint8)
            labels[allm[i] & (labels == 0)] = k
        save_mask_png(out / "masks" / f"{r.image_id}.png", union)
        (out / "instances").mkdir(exist_ok=True)
        cv2.imencode(".png", labels)[1].tofile(str(out / "instances" / f"{r.image_id}.png"))
        rows.append({"image_id": r.image_id, "split": r.split, "source_path": f"data_raw/{r.path}", "source_sha256": r.sha256,
                     "mask_path": f"masks/{r.image_id}.png", "height": h, "width": w, "detection_count": len(sel),
                     "status": status, "needs_review": status != "ok",
                     "selected_candidate_ids": [f"{r.image_id}#{i}" for i in sel],
                     "selected_mask_sha256": [mask_hash(allm[i]) for i in sel], "fg_fraction": float(union.mean()),
                     "selector_sha256": doc["selector_sha256"]})
    man = pd.DataFrame(rows)
    with open(out / "manifest.jsonl", "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/sam_candidates")
    ap.add_argument("--split", default="data/split_manifest.csv")
    ap.add_argument("--frozen", default="configs/selector_frozen.json")
    ap.add_argument("--out", default="outputs/reference_raw")
    a = ap.parse_args()
    m = make_reference(Path(a.cache), Path(a.split), Path(a.frozen), Path(a.out), Path(a.cache) / "audit.json")
    print(len(m), "images;", m.status.value_counts().to_dict(), "| detections:", int(m.detection_count.sum()))
    print(m.groupby("split").agg(n=("image_id", "size"), detections=("detection_count", "sum"), needs_review=("needs_review", "sum")))


if __name__ == "__main__":
    main()
