"""Development-only, group-aware analysis of appearance descriptors on the reviewed candidates (EXPLORATORY).

Inputs: data/review/dev_instance_verdicts.csv + data/review/verdict_verification.csv (effective verdicts),
        data/review/dev_scene_categories.csv, outputs/review_instances/review_items.csv, the SAM candidate cache, dev images.
Outputs: outputs/appearance/*.csv and experiments/appearance_analysis.md (all numbers generated, none typed by hand).
Evaluation images are rejected. Labels are AI-assisted and unverified by a human.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.analysis.appearance import (GEOMETRY, ORIENT, auc, best_threshold, cluster_bootstrap_auc, descriptors,
                                     loso_threshold)
from src.experiments.verify_verdicts import effective_verdicts
from src.segmentation.cache_candidates import unpack_masks


def md(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids the optional `tabulate` dependency)."""
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def build_table(items: pd.DataFrame, eff: pd.DataFrame, scenes: pd.DataFrame, split: pd.DataFrame, cache: Path, data: Path) -> pd.DataFrame:
    dev = set(split.loc[split["split"] == "dev", "image_id"])
    if not set(items.image_id) <= dev:
        raise ValueError("review items contain non-dev images; refusing to analyse")
    paths = dict(zip(split.image_id, split.path))
    rows, bright = [], {}
    for img_id, g in items.groupby("image_id"):
        img = cv2.imdecode(np.fromfile(str(data / paths[img_id]), np.uint8), cv2.IMREAD_COLOR)
        bright[img_id] = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
        masks = unpack_masks(cache / "masks" / f"{img_id}.npz")
        for r in g.itertuples():
            rows.append({"review_id": r.review_id, "image_id": img_id, "candidate_index": r.candidate_index,
                         **{k: getattr(r, k) for k in GEOMETRY}, "area": r.area, **descriptors(img, masks[r.candidate_index])})
    t = pd.DataFrame(rows)
    t = t.merge(eff[["review_id", "verdict", "original_verdict", "verification_status", "ambiguity_category"]], on="review_id")
    t = t.merge(scenes[["image_id", "scene"]], on="image_id")
    allb = pd.Series({i: bright[i] for i in bright})
    terc = pd.qcut(allb.rank(method="first"), 3, labels=["dim", "mid", "bright"])
    t["lighting"] = t.image_id.map(terc.astype(str))
    t["image_brightness"] = t.image_id.map(allb)
    return t


def oriented(t: pd.DataFrame, feats: dict) -> pd.DataFrame:
    return pd.DataFrame({f: t[f] * o for f, o in feats.items()})


def pooled_auc_tables(t: pd.DataFrame, feats: dict) -> pd.DataFrame:
    S = oriented(t, feats)
    out = []
    tasks = {"egg_vs_not_egg": (t.verdict == "egg", t.verdict == "not_egg"),
             "egg+bad_vs_not_egg": (t.verdict.isin(["egg", "egg_bad_boundary"]), t.verdict == "not_egg"),
             "egg_vs_bad_boundary": (t.verdict == "egg", t.verdict == "egg_bad_boundary")}
    for task, (pos, neg) in tasks.items():
        keep = (pos | neg).values
        y = pos.values[keep].astype(int)
        groups = t.scene.values[keep]
        for f in feats:
            s = S[f].values[keep]
            lo, hi = cluster_bootstrap_auc(s, y, groups) if task != "egg_vs_bad_boundary" else (np.nan, np.nan)
            out.append({"task": task, "feature": f, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                        "auc": round(auc(s[y == 1], s[y == 0]), 3), "auc_ci95_scene_bootstrap": f"{lo:.2f}-{hi:.2f}"})
    return pd.DataFrame(out)


def grouped_auc(t: pd.DataFrame, feats: dict, by: str, min_each: int = 2) -> pd.DataFrame:
    S = oriented(t, feats)
    pos = t.verdict.isin(["egg", "egg_bad_boundary"])
    neg = t.verdict == "not_egg"
    rows = []
    for g, idx in t.groupby(by).groups.items():
        p, n = pos.loc[idx], neg.loc[idx]
        row = {by: g, "n_pos": int(p.sum()), "n_neg": int(n.sum())}
        for f in feats:
            row[f] = round(auc(S.loc[idx][p][f].values, S.loc[idx][n][f].values), 3) if p.sum() >= min_each and n.sum() >= min_each else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by)


def loso_table(t: pd.DataFrame, feature_sets: dict[str, dict]) -> pd.DataFrame:
    """Leave-one-scene-out: threshold (and z-scoring) chosen on the other scenes only."""
    keep = t.verdict.isin(["egg", "egg_bad_boundary", "not_egg"]).values
    d = t[keep].reset_index(drop=True)
    y = d.verdict.isin(["egg", "egg_bad_boundary"]).astype(int).values
    groups = d.scene.values
    rows = []
    for name, feats in feature_sets.items():
        S = oriented(d, feats).values
        if S.shape[1] == 1:
            score = S[:, 0]
            pred = loso_threshold(score, y, groups)
        else:  # z-score each descriptor using ONLY the training scenes of each fold, then average
            pred = np.zeros(len(d), int)
            for g in np.unique(groups):
                te = groups == g
                mu, sd = S[~te].mean(0), S[~te].std(0) + 1e-9
                z_tr, z_te = ((S[~te] - mu) / sd).mean(1), ((S[te] - mu) / sd).mean(1)
                pred[te] = (z_te >= best_threshold(z_tr, y[~te])).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        rows.append({"feature_set": name, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
                     "balanced_acc": round(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/appearance")
    ap.add_argument("--report", default="experiments/appearance_analysis.md")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv("data/split_manifest.csv")
    items = pd.read_csv("outputs/review_instances/review_items.csv")
    verd = pd.read_csv("data/review/dev_instance_verdicts.csv", dtype=str, keep_default_na=False)
    ver = pd.read_csv("data/review/verdict_verification.csv", dtype=str, keep_default_na=False)
    eff = effective_verdicts(verd, ver)
    scenes = pd.read_csv("data/review/dev_scene_categories.csv")
    t = build_table(items, eff, scenes, split, Path("outputs/sam_candidates"), Path("data_raw"))
    t.to_csv(out / "descriptors.csv", index=False)
    app = dict(ORIENT)
    all_feats = {**app, **GEOMETRY}
    pooled = pooled_auc_tables(t, all_feats)
    pooled.to_csv(out / "auc_pooled.csv", index=False)
    scene_auc = grouped_auc(t, all_feats, "scene")
    scene_auc.to_csv(out / "auc_by_scene.csv", index=False)
    light_auc = grouped_auc(t, all_feats, "lighting")
    light_auc.to_csv(out / "auc_by_lighting.csv", index=False)
    sets = {f"{k} alone": {k: v} for k, v in app.items()}
    sets["appearance (5, z-mean)"] = app
    sets["ellipse_iou alone (geometry baseline)"] = {"ellipse_iou": +1}
    sets["geometry (4, z-mean)"] = dict(GEOMETRY)
    sets["appearance + geometry (9, z-mean)"] = all_feats
    loso = loso_table(t, sets)
    loso.to_csv(out / "loso_scene.csv", index=False)
    # sensitivity: drop the one uncertain verdict
    t_cert = t[t.verification_status != "uncertain"]
    loso_cert = loso_table(t_cert, sets)
    loso_cert.to_csv(out / "loso_scene_without_uncertain.csv", index=False)
    # hard subset = reviewed candidates S1 did NOT select: the regime in which a fallback would have to work
    prov = pd.read_csv("outputs/review_instances/review_provenance.csv")
    s1_ids = set(prov[prov.config_name == "S1_conservative"].review_id)
    t["s1_selected"] = t.review_id.isin(s1_ids)
    t.to_csv(out / "descriptors.csv", index=False)
    hard = t[~t.s1_selected].reset_index(drop=True)
    hard_pooled = pooled_auc_tables(hard, all_feats)
    hard_pooled = hard_pooled[hard_pooled.task != "egg_vs_bad_boundary"]
    hard_pooled.to_csv(out / "auc_pooled_hard_subset.csv", index=False)
    hard_loso = loso_table(hard, sets)
    hard_loso.to_csv(out / "loso_scene_hard_subset.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    lines = ["# Appearance descriptors on reviewed dev candidates (EXPLORATORY)", "",
             "Generated by `python -m src.analysis.appearance_analysis`; no number in this file is typed by hand.",
             f"Data: {len(t)} reviewed candidates on {t.image_id.nunique()} dev images, {t.scene.nunique()} scenes. "
             f"Verdicts: {t.verdict.value_counts().to_dict()} (effective verdicts; {int((t.verification_status != 'not_rechecked').sum())} re-checked, "
             f"{int((t.verification_status == 'uncertain').sum())} uncertain). Labels are AI-assisted, unverified by a human.",
             f"Hashes: items {sha('outputs/review_instances/review_items.csv')}, verdicts {sha('data/review/dev_instance_verdicts.csv')}, "
             f"verification {sha('data/review/verdict_verification.csv')}, scenes {sha('data/review/dev_scene_categories.csv')}, "
             f"split {sha('data/split_manifest.csv')}, code {sha('src/analysis/appearance.py')}/{sha('src/analysis/appearance_analysis.py')}.", "",
             "Hypotheses and directions are fixed in `src/analysis/appearance.py` before measurement. The descriptors were invented after seeing",
             "the reviewed failures, so these results are exploratory. Only the leave-one-scene-out table is out-of-scene.", "",
             "## Pooled AUC (oriented so that higher = more egg-like); interval = bootstrap over scenes", "",
             md(pooled), "",
             "## Leave-one-scene-out thresholding (threshold chosen only from the other scenes); positives = egg + egg_bad_boundary", "",
             md(loso), "",
             "Same, without the one uncertain verdict:", "", md(loso_cert), "",
             f"## Hard subset: reviewed candidates NOT selected by S1 ({len(hard)} candidates, {int(hard.verdict.isin(['egg', 'egg_bad_boundary']).sum())} egg-like)",
             "", "Pooled AUC with scene-bootstrap interval:", "", md(hard_pooled), "",
             "Leave-one-scene-out thresholding in the hard subset:", "", md(hard_loso), "",
             "## AUC by scene (egg+bad vs not_egg; NaN = fewer than 2 candidates in a class)", "", md(scene_auc), "",
             "## AUC by lighting tercile of mean image brightness", "", md(light_auc), ""]
    Path(a.report).write_text("\n".join(lines))
    print("\n".join(lines[:12]))
    print(pooled[pooled.task == "egg+bad_vs_not_egg"].to_string(index=False))
    print(loso.to_string(index=False))


if __name__ == "__main__":
    main()
