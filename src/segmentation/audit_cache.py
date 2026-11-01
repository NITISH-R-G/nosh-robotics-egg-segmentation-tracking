"""Integrity audit of the SAM candidate cache (outputs/sam_candidates).

Checks coverage, NPZ schema, mask reconstruction, manifest consistency and feature sanity.
Exits non-zero if any error is found. Errors are collected, never raised one at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.inventory import inventory
from src.segmentation.cache_candidates import FORMAT_VERSION

FEATURE_COLS = ["area", "predicted_iou", "stability_score", "solidity", "ellipse_iou"]


def audit(cache: Path, data: Path) -> dict:
    cache, data = Path(cache), Path(data)
    errors: list[str] = []
    stats: dict = {}

    run_p = cache / "run.json"
    run = None
    if not run_p.is_file():
        errors.append("run.json missing (cache run not finished?)")
    else:
        run = json.loads(run_p.read_text())
        if run.get("format_version") != FORMAT_VERSION:
            errors.append(f"format_version {run.get('format_version')} != {FORMAT_VERSION}")

    inv = inventory(data)
    inv = inv[inv.readable]
    shapes = {r.image_id: (int(r.height), int(r.width)) for r in inv.itertuples()}
    stats["n_source_images"] = len(shapes)

    jl = cache / "candidates.jsonl"
    try:
        man = pd.read_json(jl, lines=True) if jl.is_file() and jl.stat().st_size else pd.DataFrame()
    except ValueError as e:
        errors.append(f"candidates.jsonl unreadable: {e}")
        man = pd.DataFrame()
    if man.empty:
        man = pd.DataFrame(columns=["image_id", "candidate_id", "index", "mask_ref", "area", "bbox_xywh",
                                    "centroid_xy"] + FEATURE_COLS)
    dups = man.candidate_id[man.candidate_id.duplicated()].tolist()
    if dups:
        errors.append(f"duplicate candidate_ids: {dups[:5]}")
    for c in FEATURE_COLS:
        v = pd.to_numeric(man[c], errors="coerce")
        bad = int((~np.isfinite(v)).sum())
        if bad:
            errors.append(f"{bad} non-finite values in column {c}")
    unknown = set(man.image_id) - set(shapes)
    if unknown:
        errors.append(f"manifest references unknown images: {sorted(unknown)[:5]}")

    npz_dir = cache / "masks"
    npz_files = {p.stem: p for p in npz_dir.glob("*.npz")} if npz_dir.is_dir() else {}
    missing = sorted(set(shapes) - set(npz_files))
    extra = sorted(set(npz_files) - set(shapes))
    if missing:
        errors.append(f"{len(missing)} source images without npz: {missing[:5]}")
    if extra:
        errors.append(f"npz without source image: {extra[:5]}")

    per_image_n: dict[str, int] = {}
    n_checked = 0
    for img_id, shape in shapes.items():
        p = npz_files.get(img_id)
        rows = man[man.image_id == img_id].sort_values("index")
        if p is None:
            continue
        try:
            with open(p, "rb") as fh, np.load(fh) as z:  # own the handle: np.load leaks it on bad zip files
                packed, nshape = z["masks"], tuple(int(v) for v in z["shape"])
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as e:
            errors.append(f"{img_id}: unreadable npz ({type(e).__name__}: {e})")
            continue
        if nshape != shape:
            errors.append(f"{img_id}: npz shape {nshape} != image shape {shape}")
            continue
        n = packed.shape[0]
        per_image_n[img_id] = n
        h, w = shape
        if n and (packed.ndim != 2 or packed.dtype != np.uint8 or packed.shape[1] != math.ceil(h * w / 8)):
            errors.append(f"{img_id}: bad packed array {packed.shape} {packed.dtype}")
            continue
        if len(rows) != n:
            errors.append(f"{img_id}: manifest has {len(rows)} rows but npz has {n} masks")
            continue
        if n and list(rows["index"]) != list(range(n)):
            errors.append(f"{img_id}: manifest indices are not 0..{n - 1}")
            continue
        for r, row in zip(packed, rows.itertuples()):
            bits = np.unpackbits(r)
            if bits[h * w:].any():
                errors.append(f"{row.candidate_id}: non-zero padding bits")
                continue
            m = bits[: h * w].reshape(h, w)
            if int(m.sum()) != int(row.area):
                errors.append(f"{row.candidate_id}: area {int(m.sum())} != manifest {row.area}")
            if row.mask_ref != f"masks/{img_id}.npz":
                errors.append(f"{row.candidate_id}: mask_ref {row.mask_ref}")
            if m.any():
                ys, xs = np.nonzero(m)
                bx, by, bw, bh = row.bbox_xywh
                if abs(bx - xs.min()) > 1 or abs(by - ys.min()) > 1 or abs(bw - (xs.max() - xs.min() + 1)) > 1 \
                        or abs(bh - (ys.max() - ys.min() + 1)) > 1:
                    errors.append(f"{row.candidate_id}: bbox {row.bbox_xywh} inconsistent with mask")
                if abs(row.centroid_xy[0] - xs.mean()) > 1e-6 or abs(row.centroid_xy[1] - ys.mean()) > 1e-6:
                    errors.append(f"{row.candidate_id}: centroid inconsistent")
            n_checked += 1

    counts = pd.Series(per_image_n, dtype=float)
    stats.update(
        n_npz=len(npz_files), n_candidates_manifest=int(len(man)), n_masks_verified=n_checked,
        empty_candidate_images=sorted(counts[counts == 0].index.tolist()),
        candidates_per_image={"min": int(counts.min()) if len(counts) else None,
                              "median": float(counts.median()) if len(counts) else None,
                              "mean": round(float(counts.mean()), 1) if len(counts) else None,
                              "max": int(counts.max()) if len(counts) else None},
        disk_bytes_masks=sum(p.stat().st_size for p in npz_files.values()),
        disk_bytes_manifest=jl.stat().st_size if jl.is_file() else 0,
    )
    if run:
        stats.update(runtime_s=run.get("total_runtime_s"), peak_gpu_mem_mb=run.get("peak_gpu_mem_mb"),
                     generator={k: run.get(k) for k in ("points_per_side", "pred_iou_thresh", "stability_score_thresh",
                                                        "box_nms_thresh", "crop_n_layers", "min_mask_region_area")})
    stats["hashes"] = {n: hashlib.sha256((cache / n).read_bytes()).hexdigest()
                       for n in ("candidates.jsonl", "run.json") if (cache / n).is_file()}
    return {"ok": not errors, "n_errors": len(errors), "errors": errors[:50], "stats": stats}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/sam_candidates")
    ap.add_argument("--data", default="data_raw")
    ap.add_argument("--out", default="outputs/sam_candidates/audit.json")
    a = ap.parse_args()
    res = audit(Path(a.cache), Path(a.data))
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    raise SystemExit(0 if res["ok"] else 1)
