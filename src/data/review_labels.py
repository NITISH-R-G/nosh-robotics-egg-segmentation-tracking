"""Validation of the manually reviewed dev egg-count labels (data/review/dev_egg_counts*.csv).

These are image-level instance COUNT labels. They are not pixel masks, not candidate verdicts
and not ground-truth segmentation. Keep the three annotation types separate.
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COUNT_COLS = {"image_id", "expected_egg_count", "pass1_count", "annotation_process"}
REQUIRED_CORRECTION_COLS = {"image_id", "old_count", "new_count", "reason"}


def validate_counts(counts: pd.DataFrame, split: pd.DataFrame, corrections: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty list = valid)."""
    errs: list[str] = []
    missing = REQUIRED_COUNT_COLS - set(counts.columns)
    if missing:
        return [f"counts missing columns: {sorted(missing)}"]
    miss_c = REQUIRED_CORRECTION_COLS - set(corrections.columns)
    if miss_c:
        return [f"corrections missing columns: {sorted(miss_c)}"]

    dev = set(split.loc[split["split"] == "dev", "image_id"])
    evl = set(split.loc[split["split"] == "eval", "image_id"])
    ids = list(counts["image_id"])
    if len(ids) != len(set(ids)):
        errs.append(f"duplicate image_ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    if set(ids) & evl:
        errs.append(f"eval images present in labels: {sorted(set(ids) & evl)}")
    if set(ids) - dev - evl:
        errs.append(f"ids not in split manifest: {sorted(set(ids) - dev - evl)}")
    if dev - set(ids):
        errs.append(f"dev images without a label: {sorted(dev - set(ids))}")
    if len(ids) != len(dev):
        errs.append(f"{len(ids)} labels but {len(dev)} dev images")

    for col in ("expected_egg_count", "pass1_count"):
        v = pd.to_numeric(counts[col], errors="coerce")
        if v.isna().any() or ((v % 1) != 0).any() or (v < 0).any():
            errs.append(f"{col} must be non-negative integers")
    if not counts["annotation_process"].astype(str).str.contains("AI-assisted").all():
        errs.append("every row must disclose AI-assisted annotation")

    changed = counts[counts["expected_egg_count"] != counts["pass1_count"]]
    corr = corrections.set_index("image_id") if len(corrections) else corrections
    for r in changed.itertuples():
        if r.image_id not in corr.index:
            errs.append(f"{r.image_id}: count changed {r.pass1_count}->{r.expected_egg_count} without correction record")
    for r in corrections.itertuples():
        if not isinstance(r.reason, str) or not r.reason.strip():
            errs.append(f"{r.image_id}: correction has no reason")
        row = counts[counts["image_id"] == r.image_id]
        if row.empty:
            errs.append(f"{r.image_id}: correction for unknown image")
            continue
        if int(row.iloc[0]["expected_egg_count"]) != int(r.new_count) or int(row.iloc[0]["pass1_count"]) != int(r.old_count):
            errs.append(f"{r.image_id}: correction does not match counts file")
    return errs
