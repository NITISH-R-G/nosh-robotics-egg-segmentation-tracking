"""Development-only selector experiments over cached SAM candidates.

Guardrails: only images with split == 'dev' are ever loaded; the experiment log is append-only
(row_type 'result' written by this script, row_type 'decision' appended after visual inspection).
Count-based metrics compare the number of selected instances with data/review/dev_egg_counts.csv
(an unverified visual annotation of counts, not masks). A count match does not prove the right
objects were selected; overlays must be inspected.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.segmentation.cache_candidates import read_candidates, unpack_masks
from src.segmentation.masks import save_mask_png
from src.segmentation.run_sam import overlay
from src.segmentation.selector import SelectorConfig, select_from_features

_S1 = dict(min_area_frac=0.01, max_area_frac=0.60, min_solidity=0.95, max_aspect=2.0, min_ellipse_iou=0.93,
          min_pred_iou=0.90, min_stability=0.90, fallback="review")

LOG = Path("experiments/selector_log.csv")


def _code_version() -> str:
    """Hash of the selector sources (the project is not a git repo yet)."""
    import hashlib

    h = hashlib.sha256()
    for f in ("src/segmentation/selector.py", "src/experiments/selector_experiments.py"):
        h.update(Path(f).read_bytes())
    return "srchash:" + h.hexdigest()[:12]


CODE_VERSION = _code_version()
FIELDS = ["experiment_id", "row_type", "timestamp", "code_version", "config", "motivation", "dev_metrics", "runtime_s",
          "observed_failures", "decision", "next_change",
          "status", "command", "split_version", "split_sha256", "counts_sha256", "cache_run_sha256",
          "candidates_sha256", "n_dev_images"]

CONFIGS = {
    "S0_pilot_filter": SelectorConfig("S0_pilot_filter"),
    "S1_conservative": SelectorConfig(
        "S1_conservative", min_area_frac=0.01, max_area_frac=0.60, min_solidity=0.95, max_aspect=2.0,
        min_ellipse_iou=0.93, min_pred_iou=0.90, min_stability=0.90),
    "S2_conservative_border_rule": SelectorConfig(
        "S2_conservative_border_rule", min_area_frac=0.01, max_area_frac=0.60, min_solidity=0.95, max_aspect=2.0,
        min_ellipse_iou=0.93, min_pred_iou=0.90, min_stability=0.90, border_min_ellipse_iou=0.85,
        border_min_area_frac=0.05),
    "S3_S1_plus_review_flag": SelectorConfig(
        "S3_S1_plus_review_flag", min_area_frac=0.01, max_area_frac=0.60, min_solidity=0.95, max_aspect=2.0,
        min_ellipse_iou=0.93, min_pred_iou=0.90, min_stability=0.90, fallback="review"),
    "S4_S1_plus_relaxed_fallback": SelectorConfig(
        "S4_S1_plus_relaxed_fallback", min_area_frac=0.01, max_area_frac=0.60, min_solidity=0.95, max_aspect=2.0,
        min_ellipse_iou=0.93, min_pred_iou=0.90, min_stability=0.90, fallback="relaxed_top1"),
    # ---- final targeted ablations: each changes exactly one rule relative to S1 (+review flag, as S3) ----
    # E1 targets occlusion-shaped eggs (11, 12, 144): relax shape only for very confident SAM masks.
    "E1_S1_alt_shape": SelectorConfig("E1_S1_alt_shape", **_S1, alts=(SelectorConfig(
        "alt_shape", min_area_frac=0.02, max_area_frac=0.60, min_solidity=0.78, max_aspect=2.0, min_ellipse_iou=0.72,
        min_pred_iou=0.95, min_stability=0.93),)),
    # E2 targets low-confidence submerged eggs (124, 135, 145): relax confidence only for strongly egg-shaped, mid-sized masks.
    "E2_S1_alt_conf": SelectorConfig("E2_S1_alt_conf", **_S1, alts=(SelectorConfig(
        "alt_conf", min_area_frac=0.02, max_area_frac=0.10, min_solidity=0.88, max_aspect=2.0, min_ellipse_iou=0.89,
        min_pred_iou=0.65, min_stability=0.55),)),
    # E4 targets the one S1 false positive (earphone bud, image 16): drop instances far smaller than the largest.
    # E5 (follow-up derived from E2's reviewed false positives, which fall on images where S1 already found the egg):
    # apply E2's relaxed gate only when S1 selects nothing in the image.
    "E5_S1_alt_gate_fallback": SelectorConfig("E5_S1_alt_gate_fallback", **{**_S1, "fallback": "alt_gate"}, fallback_alts=(
        SelectorConfig("alt_conf", min_area_frac=0.02, max_area_frac=0.10, min_solidity=0.88, max_aspect=2.0,
                       min_ellipse_iou=0.89, min_pred_iou=0.65, min_stability=0.55),)),
    "E4_S1_rel_area": SelectorConfig("E4_S1_rel_area", **_S1, min_rel_area=0.25),
}


def _sha(p: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def append_log(row: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


_IMG_RE = re.compile(r'^\{"image_id": "([^"]+)"')
VALID_SPLITS = {"dev", "eval"}


def _full_sha(p: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require_passing_audit(cache: Path) -> dict:
    """Refuse to run unless audit.json says ok AND was computed on exactly the current cache files."""
    ap = cache / "audit.json"
    if not ap.is_file():
        raise RuntimeError(f"no audit artifact at {ap}; run python -m src.segmentation.audit_cache first")
    audit = json.loads(ap.read_text())
    if not audit.get("ok"):
        raise RuntimeError(f"cache audit did not pass ({audit.get('n_errors')} errors); refusing to run selectors")
    for name, h in audit["stats"].get("hashes", {}).items():
        if _full_sha(cache / name) != h:
            raise RuntimeError(f"stale audit: {name} changed after the audit ran; rerun the audit")
    if not audit["stats"].get("hashes"):
        raise RuntimeError("audit artifact has no content hashes; rerun the audit")
    return audit


def load_dev(cache: Path, split_csv: Path, counts_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dev-only candidates and labels. Eval rows are skipped on the raw text line, before JSON parsing.

    Split values must be exactly 'dev' or 'eval'; anything else (typos, NaN, case variants) is an error,
    never silently treated as development data.
    """
    split = pd.read_csv(split_csv)
    bad = split[~split["split"].isin(VALID_SPLITS)]
    if len(bad):
        raise ValueError(f"unknown split values {sorted(map(str, bad['split'].unique()))} in {split_csv}")
    dev_ids = set(split.loc[split["split"] == "dev", "image_id"])
    cands = read_candidates(cache, dev_ids)
    counts = pd.read_csv(counts_csv)
    stray = set(counts.image_id) - dev_ids
    if stray:
        raise ValueError(f"counts file contains non-dev images: {sorted(stray)[:5]}")
    counts = counts.sort_values("image_id", key=lambda c: c.str.extract(r"(\d+)$")[0].astype(int), kind="stable")
    return cands, counts.reset_index(drop=True)


def run_config(cfg: SelectorConfig, cands: pd.DataFrame, counts: pd.DataFrame, cache: Path, data: Path,
               paths: dict, out_dir: Path | None) -> tuple[pd.DataFrame, float]:
    t0 = time.time()
    rows = []
    for img_id, exp, tag in zip(counts.image_id, counts.expected_egg_count,
                              counts.difficulty_tag.astype(object).where(counts.difficulty_tag.notna(), "")):
        dfi = cands[cands.image_id == img_id].set_index("index")
        npz = cache / "masks" / f"{img_id}.npz"
        allm = unpack_masks(npz) if len(dfi) else np.zeros((0, 640, 640), bool)
        h, w = allm.shape[1:] if len(dfi) else (640, 640)
        sel, status = select_from_features(dfi, lambda i: allm[i], h * w, cfg)
        rows.append({"image_id": img_id, "expected": int(exp), "selected": len(sel), "status": status,
                     "tag": tag, "selected_idx": sel})
        if out_dir is not None:
            union = np.zeros((h, w), np.uint8)
            labels = np.zeros((h, w), np.uint8)
            for k, i in enumerate(sel, 1):
                union |= allm[i].astype(np.uint8)
                labels[allm[i] & (labels == 0)] = k
            save_mask_png(out_dir / "masks" / f"{img_id}.png", union)
            img = cv2.imdecode(np.fromfile(str(data / paths[img_id]), np.uint8), cv2.IMREAD_COLOR)
            (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
            ov = overlay(img, labels)
            cv2.putText(ov, f"{img_id[-3:]} exp={exp} sel={len(sel)} {status}", (6, 24), 0, 0.7, (0, 255, 255), 2)
            cv2.imencode(".jpg", ov)[1].tofile(str(out_dir / "overlays" / f"{img_id}.jpg"))
    return pd.DataFrame(rows), time.time() - t0


def summarize(res: pd.DataFrame) -> dict:
    def block(d: pd.DataFrame) -> dict:
        diff = d.selected - d.expected
        a = diff.abs()
        return {"n_images": int(len(d)), "expected_eggs": int(d.expected.sum()), "selected": int(d.selected.sum()),
                "count_exact_rate": round(float((diff == 0).mean()), 3), "over_rate": round(float((diff > 0).mean()), 3),
                "under_rate": round(float((diff < 0).mean()), 3),
                "empty_rate": round(float((d.selected == 0).mean()), 3),
                "mean_abs_count_err": round(float(a.mean()), 3), "median_abs_count_err": float(a.median()),
                "max_abs_count_err": int(a.max()),
                "review_flagged": int((d.status == "review_no_candidate").sum()),
                "fallback_used": int((d.status == "fallback_top1").sum())}
    by_tag = {(t if t else "untagged"): block(g) for t, g in res.groupby(res.tag.replace("", "untagged"))}
    return {"all_dev": block(res), "excluding_submerged": block(res[res.tag != "submerged"]), "by_tag": by_tag}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--cache", default="outputs/sam_candidates")
    ap.add_argument("--data", default="data_raw")
    ap.add_argument("--split", default="data/split_manifest.csv")
    ap.add_argument("--counts", default="data/review/dev_egg_counts.csv")
    ap.add_argument("--motivation", default="")
    ap.add_argument("--save", action="store_true", help="write dev masks and overlays")
    a = ap.parse_args()
    cache, data = Path(a.cache), Path(a.data)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    resolved = json.dumps(vars(a), sort_keys=True)
    pre = {"experiment_id": f"preflight@{stamp}", "row_type": "result", "timestamp": stamp,
           "code_version": CODE_VERSION, "command": "python -m src.experiments.selector_experiments " + resolved}
    try:
        audit = require_passing_audit(cache)
        cands, counts = load_dev(cache, Path(a.split), Path(a.counts))
        sp = pd.read_csv(a.split)
        paths = dict(zip(sp.image_id, sp.path))
        meta = {"command": pre["command"],
                "split_version": "|".join(sorted(set(sp.split_version.astype(str)))) if "split_version" in sp else "unknown",
                "split_sha256": _full_sha(Path(a.split))[:16], "counts_sha256": _full_sha(Path(a.counts))[:16],
                "cache_run_sha256": audit["stats"]["hashes"]["run.json"][:16],
                "candidates_sha256": audit["stats"]["hashes"]["candidates.jsonl"][:16],
                "n_dev_images": int(len(counts))}
    except Exception as e:  # preflight failures are logged too
        append_log({**pre, "status": f"failed_preflight: {type(e).__name__}: {e}"})
        raise
    for name in a.configs:
        cfg = CONFIGS[name]
        eid = f"{name}@{stamp}"
        base = {"experiment_id": eid, "row_type": "result", "timestamp": stamp, "code_version": CODE_VERSION,
                "config": json.dumps(cfg.as_dict()), "motivation": a.motivation, **meta}
        out_dir = Path("outputs/selector_dev") / name if a.save else None
        try:
            res, secs = run_config(cfg, cands, counts, cache, data, paths, out_dir)
        except Exception as e:  # failed trials are logged, then re-raised
            append_log({**base, "status": f"failed: {type(e).__name__}: {e}"})
            raise
        m = summarize(res)
        fails = res[res.selected != res.expected]
        append_log({**base, "dev_metrics": json.dumps(m), "runtime_s": round(secs, 2), "status": "completed",
                    "observed_failures": "count mismatch ids: " + ",".join(
                        f"{r.image_id[-3:]}({r.expected}->{r.selected})" for r in fails.itertuples())})
        Path("experiments").mkdir(exist_ok=True)
        res.assign(selected_idx=res.selected_idx.map(lambda v: json.dumps([int(i) for i in v])),
                   config_name=name, experiment_id=eid, code_version=CODE_VERSION).to_csv(
            f"experiments/selector_{name}_{stamp}_per_image.csv", index=False)
        print(eid, json.dumps(m["all_dev"]))


if __name__ == "__main__":
    main()
