"""Load the two development references (kept separate) and reconstruct their instance lists.

R_raw: outputs/selector_dev/S1_conservative/masks (union of the S1-selected candidates). R_rev: outputs/reference_reviewed_dev/masks
(R_raw plus the manual additions). Neither stores instance labels, so instances are the individual cached candidate masks
(S1 `selected_idx`, plus the manifest additions for R_rev). The union of the reconstructed instances must equal the stored mask bit for bit.
Both are pseudo-ground-truth derived from SAM; labels are AI-assisted and unverified. Evaluation images are rejected.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.diagnose import assert_dev
from src.segmentation.cache_candidates import unpack_masks
from src.segmentation.masks import load_mask_png

ADD = "add_to_reviewed_reference"


def load_references(dev_ids: list[str], split_csv: Path, raw_dir: Path, rev_dir: Path, s1_run_csv: Path, manifest_csv: Path,
                    cache: Path) -> dict[str, dict]:
    assert_dev(list(dev_ids), Path(split_csv))
    run = pd.read_csv(s1_run_csv).set_index("image_id")
    adds = pd.read_csv(manifest_csv)
    adds = adds[adds.action == ADD]
    out = {}
    for img in dev_ids:
        masks = unpack_masks(Path(cache) / "masks" / f"{img}.npz")
        raw_idx = json.loads(run.loc[img, "selected_idx"])
        add_idx = [int(c.split("#")[1]) for c in adds[adds.image_id == img].candidate_id]
        raw_union = load_mask_png(Path(raw_dir) / f"{img}.png", masks.shape[1:]).astype(bool)
        rev_union = load_mask_png(Path(rev_dir) / f"{img}.png", masks.shape[1:]).astype(bool)
        raw_inst = [masks[i] for i in raw_idx]
        rev_inst = raw_inst + [masks[i] for i in add_idx]
        for name, inst, stored in (("raw", raw_inst, raw_union), ("reviewed", rev_inst, rev_union)):
            u = np.logical_or.reduce(inst) if inst else np.zeros(masks.shape[1:], bool)
            if not (u == stored).all():
                raise ValueError(f"{img}: reconstructed {name} instances do not reproduce the stored {name} mask")
        out[img] = {"raw_union": raw_union, "rev_union": rev_union, "raw_instances": raw_inst, "rev_instances": rev_inst,
                    "raw_candidate_ids": [f"{img}#{i}" for i in raw_idx], "rev_candidate_ids": [f"{img}#{i}" for i in raw_idx + add_idx]}
    return out
