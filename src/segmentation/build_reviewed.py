"""Build the manually reviewed dev reference masks in a directory SEPARATE from the raw automatic masks.

reviewed mask = raw S1 mask OR the cached masks of the manifest rows marked `add_to_reviewed_reference`.
The raw masks are never written (their bytes are hashed before and after), every added candidate is verified against the cache by
identity and mask hash, evaluation images are rejected, and an existing output is never overwritten.
Labels behind the manifest are AI-assisted and not verified by a human.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.diagnose import assert_dev
from src.experiments.review_pool import mask_sha256, verify_identity
from src.segmentation.cache_candidates import read_candidates, unpack_masks
from src.segmentation.masks import load_mask_png, save_mask_png

ADD = "add_to_reviewed_reference"


def _dir_digest(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(d).glob("*.png")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def build_reviewed(manifest: pd.DataFrame, cache: Path, raw_dir: Path, out_dir: Path, split_csv: Path) -> pd.DataFrame:
    cache, raw_dir, out_dir = Path(cache), Path(raw_dir), Path(out_dir)
    if out_dir.resolve() == raw_dir.resolve() or raw_dir.resolve() in out_dir.resolve().parents or cache.resolve() in out_dir.resolve().parents:
        raise ValueError("reviewed masks must be written to a directory separate from the raw masks and the candidate cache")
    if (out_dir / "reviewed_manifest.jsonl").exists():
        raise FileExistsError(f"{out_dir / 'reviewed_manifest.jsonl'} exists; refusing to overwrite a reviewed reference")
    sp = pd.read_csv(split_csv)
    dev_ids = sorted(sp.loc[sp["split"] == "dev", "image_id"])
    assert_dev(sorted(manifest.image_id.unique()), Path(split_csv))
    adds = manifest[manifest.action == ADD]
    if adds.duplicated(["image_id", "candidate_id"]).any():
        raise ValueError("duplicate additions in the manifest")
    before = _dir_digest(raw_dir)
    cands = read_candidates(cache, set(adds.image_id))
    rows = []
    for img in dev_ids:
        raw = load_mask_png(raw_dir / f"{img}.png")
        rev = raw.copy()
        added = []
        for r in adds[adds.image_id == img].itertuples():
            masks = unpack_masks(cache / "masks" / f"{img}.npz")
            idx = int(r.candidate_id.split("#")[1])
            verify_identity(cands, masks, img, idx)
            if mask_sha256(masks[idx]) != r.mask_sha256:
                raise ValueError(f"{r.candidate_id}: cached mask hash differs from the manifest; refusing to add it")
            if masks[idx].shape != rev.shape:
                raise ValueError(f"{r.candidate_id}: shape mismatch with the raw mask")
            rev = rev | masks[idx].astype(np.uint8)
            added.append(r.candidate_id)
        save_mask_png(out_dir / "masks" / f"{img}.png", rev)
        rows.append({"image_id": img, "added_candidate_ids": added, "raw_fg_pixels": int(raw.sum()), "reviewed_fg_pixels": int(rev.sum()),
                     "raw_mask_sha256": mask_sha256(raw), "reviewed_mask_sha256": mask_sha256(rev), "changed": bool(added)})
    if _dir_digest(raw_dir) != before:
        raise RuntimeError("raw masks changed during the build (this must never happen)")
    man = pd.DataFrame(rows)
    with open(out_dir / "reviewed_manifest.jsonl", "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/review/dev_manual_selection_manifest.csv")
    ap.add_argument("--cache", default="outputs/sam_candidates")
    ap.add_argument("--raw", default="outputs/selector_dev/S1_conservative/masks")
    ap.add_argument("--split", default="data/split_manifest.csv")
    ap.add_argument("--out", default="outputs/reference_reviewed_dev")
    a = ap.parse_args()
    man = build_reviewed(pd.read_csv(a.manifest), Path(a.cache), Path(a.raw), Path(a.out), Path(a.split))
    print(len(man), "dev images;", int(man.changed.sum()), "changed by manual additions;",
          int(man.reviewed_fg_pixels.sum() - man.raw_fg_pixels.sum()), "foreground pixels added in total")


if __name__ == "__main__":
    main()
