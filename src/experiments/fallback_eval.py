"""Phase 3: does an appearance-scored fallback improve S1 in the images where S1 selects nothing? (dev only, exploratory)

Pre-registered before any number was computed (experiments/fallback_evaluation.md):
  pool     = NMS survivors of the union of two relaxed gates (alt_shape, alt_conf), only in images where S1 selects nothing
  score    = mean z-score of the oriented descriptors L_in, dL_ring, texture_in (pooled AUC >= 0.8 in the Phase 2 pooled table);
             z-scoring uses every reviewed dev candidate EXCEPT the held-out image
  rule     = accept pool candidates whose score >= tau; tau maximises balanced accuracy on the pool candidates of the OTHER images
  accept   = >= 5 new egg-like detections, <= 1 new false positive, >= 60% of new detections clean `egg`, S1 selections unchanged
Every accepted candidate is traceable by candidate id and mask hash. Verdicts are AI-assisted and unverified by a human.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.appearance import ORIENT, best_threshold
from src.experiments.review_pool import mask_sha256
from src.experiments.selector_experiments import load_dev
from src.experiments.verify_verdicts import effective_verdicts
from src.segmentation.cache_candidates import unpack_masks
from src.segmentation.selector import SelectorConfig, select_from_features

SCORE_FEATURES = {k: ORIENT[k] for k in ("L_in", "dL_ring", "texture_in")}
ALT_SHAPE = SelectorConfig("alt_shape", min_area_frac=0.02, max_area_frac=0.60, min_solidity=0.78, max_aspect=2.0,
                           min_ellipse_iou=0.72, min_pred_iou=0.95, min_stability=0.93)
ALT_CONF = SelectorConfig("alt_conf", min_area_frac=0.02, max_area_frac=0.10, min_solidity=0.88, max_aspect=2.0,
                          min_ellipse_iou=0.89, min_pred_iou=0.65, min_stability=0.55)
POOL_CFG = SelectorConfig("pool", min_area_frac=0.01, max_area_frac=0.60, min_solidity=0.95, max_aspect=2.0,
                          min_ellipse_iou=0.93, min_pred_iou=0.90, min_stability=0.90, fallback="alt_gate",
                          fallback_alts=(ALT_SHAPE, ALT_CONF))
CRITERIA = {"min_new_egg_like": 5, "max_new_false_positives": 1, "min_clean_egg_share": 0.60}


def z_score(df: pd.DataFrame, mu: pd.Series, sd: pd.Series) -> np.ndarray:
    """Mean over descriptors of the oriented z-score (higher = more egg-like)."""
    return np.mean([(df[f].values - mu[f]) / sd[f] * o for f, o in SCORE_FEATURES.items()], axis=0)


def build_pool(cands: pd.DataFrame, image_ids, cache: Path, s1_selected: dict[str, list[int]]) -> pd.DataFrame:
    rows = []
    for img in image_ids:
        if s1_selected.get(img):  # S1 selected something: the fallback never applies
            continue
        dfi = cands[cands.image_id == img].set_index("index", drop=False)
        masks = unpack_masks(cache / "masks" / f"{img}.npz")
        sel, status = select_from_features(dfi, lambda i: masks[i], masks.shape[1] * masks.shape[2], POOL_CFG)
        for i in sel:
            rows.append({"image_id": img, "candidate_index": int(i), "candidate_id": f"{img}#{i}",
                         "mask_sha256": mask_sha256(masks[i]), "pool_status": status})
    return pd.DataFrame(rows)


def evaluate(pool: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-image-out. `table` = every reviewed dev candidate with descriptors, verdict and mask_sha256."""
    d = pool.merge(table.drop(columns=["candidate_index"]), on=["image_id", "mask_sha256"], how="left")
    if d.verdict.isna().any():
        raise ValueError(f"unreviewed pool candidates: {d[d.verdict.isna()].candidate_id.tolist()}")
    d["egg_like"] = d.verdict.isin(["egg", "egg_bad_boundary"]).astype(int)
    out = []
    for img in sorted(d.image_id.unique()):
        ref = table[table.image_id != img]  # z-scoring statistics never see the held-out image
        mu, sd = ref[list(SCORE_FEATURES)].mean(), ref[list(SCORE_FEATURES)].std()
        tr = d[d.image_id != img]
        tau = best_threshold(z_score(tr, mu, sd), tr.egg_like.values) if tr.egg_like.nunique() > 1 else float("inf")
        cur = d[d.image_id == img].copy()
        cur["score"] = z_score(cur, mu, sd)
        cur["tau_from_other_images"] = tau
        cur["accepted"] = cur.score >= tau
        out.append(cur)
    return pd.concat(out, ignore_index=True)


def summarize(res: pd.DataFrame) -> dict:
    acc = res[res.accepted]
    tp, fp = int(acc.egg_like.sum()), int((1 - acc.egg_like).sum())
    clean = int((acc.verdict == "egg").sum())
    share = (clean / len(acc)) if len(acc) else None
    return {"new_detections": int(len(acc)), "new_egg_like": tp, "new_clean_egg": clean, "new_bad_boundary": tp - clean,
            "new_false_positives": fp, "clean_share": None if share is None else round(share, 3),
            "pool_candidates": int(len(res)), "pool_egg_like": int(res.egg_like.sum()),
            "criteria_met": bool(share is not None and tp >= CRITERIA["min_new_egg_like"]
                                 and fp <= CRITERIA["max_new_false_positives"] and share >= CRITERIA["min_clean_egg_share"])}


def mask_overlap_duplicates(cache: Path, rows: pd.DataFrame) -> int:
    """Pairs of selected masks in the same image that overlap enough to be the same egg (NMS rule: IoU > 0.5 or containment > 0.8)."""
    n = 0
    for img, g in rows.groupby("image_id"):
        if len(g) < 2:
            continue
        masks = unpack_masks(cache / "masks" / f"{img}.npz")
        idx = [int(c.split("#")[1]) for c in g.candidate_id]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b = masks[idx[i]], masks[idx[j]]
                inter = float((a & b).sum())
                if inter / float((a | b).sum()) > 0.5 or inter / float(min(a.sum(), b.sum())) > 0.8:
                    n += 1
    return n


def config_metrics(name: str, sel: pd.DataFrame, counts: pd.DataFrame, cache: Path) -> tuple[dict, pd.DataFrame]:
    """sel: one row per selected candidate (image_id, candidate_id, verdict). Metrics over all 88 dev images."""
    s = sel.assign(lenient=sel.verdict.isin(["egg", "egg_bad_boundary"]), strict=sel.verdict == "egg", fp=sel.verdict == "not_egg")
    per = s.groupby("image_id").agg(selected=("candidate_id", "size"), tp_lenient=("lenient", "sum"), tp_strict=("strict", "sum"),
                                    fp=("fp", "sum"))
    m = counts.set_index("image_id")[["expected_egg_count"]].join(per).fillna(0).astype(int)
    m["abs_err"] = (m.selected - m.expected_egg_count).abs()
    m["missed"] = (m.expected_egg_count - m.tp_lenient).clip(lower=0)
    n_eggs, tp, tps, fp = int(m.expected_egg_count.sum()), int(m.tp_lenient.sum()), int(m.tp_strict.sum()), int(m.fp.sum())
    diff = m.selected - m.expected_egg_count
    row = {"config": name, "images": len(m), "labelled_eggs": n_eggs, "selected": int(m.selected.sum()),
           "count_exact_rate": round(float((diff == 0).mean()), 3), "over_rate": round(float((diff > 0).mean()), 3),
           "under_rate": round(float((diff < 0).mean()), 3), "mean_abs_err": round(float(m.abs_err.mean()), 3),
           "median_abs_err": float(m.abs_err.median()), "max_abs_err": int(m.abs_err.max()),
           "tp_lenient": tp, "tp_strict": tps, "bad_boundary_tp": tp - tps, "false_positives": fp, "missed_eggs": int(m.missed.sum()),
           "precision_lenient": round(tp / max(tp + fp, 1), 3), "recall_lenient": round(tp / n_eggs, 3),
           "precision_strict": round(tps / max(tp + fp, 1), 3), "recall_strict": round(tps / n_eggs, 3),
           "images_with_fp": int((m.fp > 0).sum()), "duplicate_pairs": mask_overlap_duplicates(cache, sel)}
    return row, m.reset_index().assign(config=name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-run", required=True, help="per-image CSV of the S1 run")
    ap.add_argument("--out", default="experiments")
    a = ap.parse_args()
    cache = Path("outputs/sam_candidates")
    cands, counts = load_dev(cache, Path("data/split_manifest.csv"), Path("data/review/dev_egg_counts.csv"))
    s1 = pd.read_csv(a.s1_run)
    s1_sel = {r.image_id: json.loads(r.selected_idx) for r in s1.itertuples()}
    items = pd.read_csv("outputs/review_instances/review_items.csv")
    verd = pd.read_csv("data/review/dev_instance_verdicts.csv", dtype=str, keep_default_na=False)
    ver = pd.read_csv("data/review/verdict_verification.csv", dtype=str, keep_default_na=False)
    eff = effective_verdicts(verd, ver)[["review_id", "verdict", "verification_status"]]
    desc = pd.read_csv("outputs/appearance/descriptors.csv")[["review_id", "candidate_index", *SCORE_FEATURES]]
    table = items[["review_id", "image_id", "mask_sha256"]].merge(eff, on="review_id").merge(desc, on="review_id")
    pool = build_pool(cands, counts.image_id, cache, s1_sel)
    res = evaluate(pool, table)
    res.to_csv(Path(a.out) / "fallback_pool_scores.csv", index=False)
    summ = summarize(res)
    print(json.dumps(summ, indent=1))
    print(res[["image_id", "candidate_id", "verdict", "score", "tau_from_other_images", "accepted"]].round(2).to_string(index=False))
    Path(a.out, "fallback_summary.json").write_text(json.dumps({"criteria": CRITERIA, "summary": summ}, indent=1))

    # ---- S1 alone vs S1 + pool-all (no appearance) vs S1 + appearance fallback, identical dev images and verdicts
    prov = pd.read_csv("outputs/review_instances/review_provenance.csv")
    v = eff.set_index("review_id").verdict
    s1_rows = prov[prov.config_name == "S1_conservative"].drop_duplicates("candidate_id")[["image_id", "candidate_id", "review_id", "mask_sha256"]]
    s1_rows = s1_rows.assign(verdict=s1_rows.review_id.map(v))
    fb_all = res.assign(verdict=res.verdict)[["image_id", "candidate_id", "mask_sha256", "verdict"]]
    fb_acc = res[res.accepted][["image_id", "candidate_id", "mask_sha256", "verdict"]]
    base = s1_rows[["image_id", "candidate_id", "mask_sha256", "verdict"]]
    # S1 selections must be untouched by construction: the fallback only adds rows in images where S1 selected nothing
    assert not set(fb_acc.image_id) & set(base.image_id), "fallback touched an image where S1 selected something"
    rows, per = [], []
    for name, sel in (("S1 alone", base), ("S1 + pool-all (no appearance score)", pd.concat([base, fb_all])),
                      ("S1 + appearance fallback (pre-registered rule)", pd.concat([base, fb_acc]))):
        r, p = config_metrics(name, sel, counts, cache)
        rows.append(r)
        per.append(p)
    comp = pd.DataFrame(rows)
    comp.to_csv(Path(a.out) / "fallback_comparison.csv", index=False)
    pd.concat(per).to_csv(Path(a.out) / "fallback_per_image.csv", index=False)
    trace = pd.concat([base.assign(source="S1", score=np.nan, accepted=True),
                       res[["image_id", "candidate_id", "mask_sha256", "verdict", "score", "accepted"]].assign(source="fallback_pool")],
                      ignore_index=True)
    trace.to_csv(Path(a.out) / "fallback_selections.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(comp.T.to_string())


if __name__ == "__main__":
    main()
