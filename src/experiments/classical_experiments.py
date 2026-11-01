"""Pre-registered classical experiments E00-E13 (experiments/classical_comparison_protocol_amendment_1.md), development images only.

Everything that decides anything is implemented once here and tested: per-image evaluation against R_raw and R_rev SEPARATELY, aggregation,
the keep/discard rule (A6), the greedy sequence, and leave-one-scene-out re-runs of the whole sequence. The closed experiment list cannot be
extended at run time. Labels and both references are AI-assisted / SAM-derived pseudo-ground-truth: results are agreement, not accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.evaluation.metrics import count_metrics, match_instances, pixel_metrics, prf
from src.evaluation.references import load_references
from src.experiments.diagnose import assert_dev
from src.segmentation.classical import ClassicalConfig, segment_classical
from src.segmentation.masks import save_mask_png

LOG = Path("experiments/classical_log.csv")
LOG_FIELDS = ["experiment_id", "timestamp", "code_version", "dataset_version", "split_seed", "preprocessing", "color_space", "threshold_method",
              "parameter_values", "morphology", "component_filters", "motivation", "development_metrics", "runtime", "observed_failures",
              "decision", "next_change"]

# (id, change relative to the comparison base, motivation, kind) -- CLOSED LIST (amendment A5)
EXPERIMENTS: list[tuple[str, dict, str, str]] = [
    ("E00", {}, "baseline: simplest interpretable chain (Lab lightness, Otsu, 5 px open/close, fill holes, area 0.5-60%)", "base"),
    ("E01", {"colour": "V"}, "HSV value as the foreground map: less affected by chroma than Lab L?", "colour"),
    ("E02", {"colour": "W"}, "whiteness (L - chroma): eggs are bright AND unsaturated; should reject coloured bright regions (cloth, bowls, tile)", "colour"),
    ("E03", {"colour": "S"}, "inverted saturation: eggs are nearly unsaturated", "colour"),
    ("E04", {"blur_sigma": 2.0}, "blur before Otsu to suppress texture (cloth, rubble) that splits regions", "single"),
    ("E05", {"threshold": "adaptive", "adaptive_block": 151, "adaptive_c": -5.0}, "local threshold for uneven lighting and gradients", "single"),
    ("E06", {"open_k": 9, "close_k": 9}, "larger morphology to remove bridges to background and fill gaps", "single"),
    ("E07", {"open_k": 3, "close_k": 3}, "smaller morphology to keep thin egg boundaries", "single"),
    ("E08", {"min_solidity": 0.85}, "egg-like shapes are convex: reject ragged regions", "single"),
    ("E09", {"max_aspect": 2.2}, "eggs are compact: reject elongated bars", "single"),
    ("E10", {"min_area_frac": 0.01}, "reject larger specks and small background blobs", "single"),
    ("E11", {"border_min_area_frac": 0.05}, "border-touching components must be large (clipped eggs are large)", "single"),
    ("E12", {"split_watershed": True, "split_h": 0.4}, "separate touching eggs with a distance-transform watershed", "single"),
    ("E13", {"min_ellipse_iou": 0.85}, "shape prior: component must resemble its fitted ellipse", "single"),
]
RULE = {"min_gain_f1": 0.02, "max_loss_iou": 0.01, "max_loss_exact": 0.03}  # amendment A6


# ---------------------------------------------------------------- evaluation
def evaluate_config(cfg: ClassicalConfig, inputs: dict, save_dir: Path | None = None) -> pd.DataFrame:
    """Run the pipeline on every dev image and score it against R_raw and R_rev separately. One row per image."""
    rows = []
    for img_id in inputs["ids"]:
        img, ref = inputs["images"][img_id], inputs["refs"][img_id]
        t0 = time.perf_counter()
        union, inst, info = segment_classical(img, cfg)
        dt = time.perf_counter() - t0
        row = {"image_id": img_id, "scene": inputs["scene"][img_id], "tag": inputs["tag"][img_id], "expected": inputs["expected"][img_id],
               "n_pred": len(inst), "runtime_s": dt, "pred_fg_px": int(union.sum())}
        for name, u, ii in (("raw", ref["raw_union"], ref["raw_instances"]), ("rev", ref["rev_union"], ref["rev_instances"])):
            pm = pixel_metrics(u, union)
            mt = match_instances(inst, ii, 0.5)
            row.update({f"{name}_ref_empty": pm["ref_empty"], f"{name}_pred_empty": pm["pred_empty"], f"{name}_iou": pm["iou"],
                        f"{name}_dice": pm["dice"], f"{name}_tp_px": pm["tp"], f"{name}_fp_px": pm["fp"], f"{name}_fn_px": pm["fn"],
                        f"{name}_n_ref_inst": len(ii), f"{name}_inst_tp": mt["tp"], f"{name}_inst_fp": mt["fp"], f"{name}_inst_fn": mt["fn"]})
        rows.append(row)
        if save_dir is not None:
            save_mask_png(Path(save_dir) / "masks" / f"{img_id}.png", union.astype(bool))
    return pd.DataFrame(rows)


def evaluate_s1(inputs: dict) -> pd.DataFrame:
    """S1 scored with the SAME functions: its instances are the raw reference instances (context only: R_raw is S1)."""
    rows = []
    for img_id in inputs["ids"]:
        ref = inputs["refs"][img_id]
        row = {"image_id": img_id, "scene": inputs["scene"][img_id], "tag": inputs["tag"][img_id], "expected": inputs["expected"][img_id],
               "n_pred": len(ref["raw_instances"]), "runtime_s": float("nan"), "pred_fg_px": int(ref["raw_union"].sum())}
        for name, u, ii in (("raw", ref["raw_union"], ref["raw_instances"]), ("rev", ref["rev_union"], ref["rev_instances"])):
            pm = pixel_metrics(u, ref["raw_union"])
            mt = match_instances(ref["raw_instances"], ii, 0.5)
            row.update({f"{name}_ref_empty": pm["ref_empty"], f"{name}_pred_empty": pm["pred_empty"], f"{name}_iou": pm["iou"],
                        f"{name}_dice": pm["dice"], f"{name}_tp_px": pm["tp"], f"{name}_fp_px": pm["fp"], f"{name}_fn_px": pm["fn"],
                        f"{name}_n_ref_inst": len(ii), f"{name}_inst_tp": mt["tp"], f"{name}_inst_fp": mt["fp"], f"{name}_inst_fn": mt["fn"]})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, ids: set[str] | None = None) -> dict:
    """Metrics over a subset of images; each reference separately. Macro = mean over images with a non-empty reference."""
    d = df if ids is None else df[df.image_id.isin(ids)]
    out = {"n_images": int(len(d))}
    for name in ("raw", "rev"):
        ne = d[~d[f"{name}_ref_empty"]]
        tp, fp, fn = int(d[f"{name}_tp_px"].sum()), int(d[f"{name}_fp_px"].sum()), int(d[f"{name}_fn_px"].sum())
        itp, ifp, ifn = int(d[f"{name}_inst_tp"].sum()), int(d[f"{name}_inst_fp"].sum()), int(d[f"{name}_inst_fn"].sum())
        out[name] = {"macro_iou": float(ne[f"{name}_iou"].mean()) if len(ne) else float("nan"),
                     "macro_dice": float(ne[f"{name}_dice"].mean()) if len(ne) else float("nan"),
                     "pooled_iou": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
                     "pooled_dice": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan"),
                     "pooled_precision": tp / (tp + fp) if tp + fp else float("nan"), "pooled_recall": tp / (tp + fn) if tp + fn else float("nan"),
                     "n_images_ref_nonempty": int(len(ne)), "n_images_ref_empty": int(d[f"{name}_ref_empty"].sum()),
                     "images_ref_empty_with_prediction": int((d[f"{name}_ref_empty"] & ~d[f"{name}_pred_empty"]).sum()),
                     "fp_px_in_ref_empty_images": int(d.loc[d[f"{name}_ref_empty"], f"{name}_fp_px"].sum()),
                     "images_empty_prediction_nonempty_ref": int((~d[f"{name}_ref_empty"] & d[f"{name}_pred_empty"]).sum()),
                     "inst_tp": itp, "inst_fp": ifp, "inst_fn": ifn, **{f"inst_{k}": v for k, v in prf(itp, ifp, ifn).items()}}
    out["counts"] = count_metrics(d.n_pred.values, d.expected.values)
    out["runtime_median_s"] = float(d.runtime_s.median()) if d.runtime_s.notna().any() else float("nan")
    out["runtime_total_s"] = float(d.runtime_s.sum()) if d.runtime_s.notna().any() else float("nan")
    return out


def key_scores(df: pd.DataFrame, ids: set[str] | None = None) -> dict:
    a = aggregate(df, ids)
    return {"f1_rev": a["rev"]["inst_f1"], "iou_rev": a["rev"]["macro_iou"], "exact": a["counts"]["exact_rate"]}


def decide(base: dict, trial: dict, rule: dict = RULE) -> tuple[bool, dict]:
    """Amendment A6: keep only if F1 (vs R_rev) gains >= 0.02, macro IoU (vs R_rev) loses <= 0.01, exact-count rate loses <= 0.03."""
    d = {k: trial[k] - base[k] for k in base}
    keep = bool(d["f1_rev"] >= rule["min_gain_f1"] - 1e-12 and d["iou_rev"] >= -rule["max_loss_iou"] - 1e-12 and d["exact"] >= -rule["max_loss_exact"] - 1e-12)
    return keep, d


def run_sequence(eval_fn, ids: set[str], experiments=EXPERIMENTS) -> tuple[ClassicalConfig, list[dict]]:
    """Greedy sequence of the closed list: E00 baseline; colour spaces judged against E00 (best kept one carried on); then each single change
    is judged against the current best. eval_fn(cfg) -> per-image DataFrame (cacheable). Returns final config and the step records."""
    steps, current = [], None
    colour_trials = []
    for eid, change, why, kind in experiments:
        if kind == "base":
            current = ClassicalConfig(**change)
            steps.append({"id": eid, "cfg": current, "decision": "baseline", "scores": key_scores(eval_fn(current), ids), "deltas": None})
            base_scores = steps[-1]["scores"]
            continue
        trial = replace(current, **change)
        sc = key_scores(eval_fn(trial), ids)
        if kind == "colour":
            keep, d = decide(base_scores, sc)
            colour_trials.append((eid, trial, sc, d, keep))
            steps.append({"id": eid, "cfg": trial, "decision": "pending_colour_choice", "scores": sc, "deltas": d})
            nxt = [e for e in experiments if e[0] > eid][:1]
            if not nxt or nxt[0][3] != "colour":  # last colour trial: choose the best kept one
                kept = [t for t in colour_trials if t[4]]
                best = max(kept, key=lambda t: t[2]["f1_rev"]) if kept else None
                for s in steps:
                    if s["decision"] == "pending_colour_choice":
                        s["decision"] = "kept" if best and s["id"] == best[0] else "discarded"
                if best:
                    current, base_scores = best[1], best[2]
            continue
        keep, d = decide(base_scores, sc)
        steps.append({"id": eid, "cfg": trial, "decision": "kept" if keep else "discarded", "scores": sc, "deltas": d})
        if keep:
            current, base_scores = trial, sc
    return current, steps


def loso(eval_fn, inputs: dict, experiments=EXPERIMENTS) -> pd.DataFrame:
    """Re-run the whole sequence 15 times, each time without one scene; evaluate that configuration on the held-out scene only."""
    rows = []
    scenes = sorted(set(inputs["scene"].values()))
    for sc in scenes:
        held = {i for i in inputs["ids"] if inputs["scene"][i] == sc}
        cfg, steps = run_sequence(eval_fn, set(inputs["ids"]) - held, experiments)
        df = eval_fn(cfg)
        for name in ("raw", "rev"):
            d = df[df.image_id.isin(held)]
            rows.append({"held_out_scene": sc, "reference": name, "n_images": len(held), "config_sha": cfg.sha(),
                         "kept": ",".join(s["id"] for s in steps if s["decision"] in ("kept", "baseline")),
                         "inst_tp": int(d[f"{name}_inst_tp"].sum()), "inst_fp": int(d[f"{name}_inst_fp"].sum()), "inst_fn": int(d[f"{name}_inst_fn"].sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- inputs / logging
def subsets(ids: list[str], manifest_csv: Path) -> dict[str, set[str]]:
    man = pd.read_csv(manifest_csv)
    first_pass = set(man[(man.action == "add_to_reviewed_reference") & (man.verification_status == "first_pass_only")].image_id)
    unresolved = set(man[man.action == "exclude"].image_id)
    added = set(man[man.action == "add_to_reviewed_reference"].image_id)
    s = set(ids)
    return {"S_all": s, "S_no_first_pass": s - first_pass, "S_clean": s - added - unresolved}


def load_inputs(ids_file="experiments/classical_dev_image_ids.txt", split_csv="data/split_manifest.csv", data="data_raw") -> dict:
    ids = Path(ids_file).read_text().split()
    assert_dev(ids, Path(split_csv))
    sp = pd.read_csv(split_csv).set_index("image_id")
    counts = pd.read_csv("data/review/dev_egg_counts.csv").set_index("image_id")
    scenes = pd.read_csv("data/review/dev_scene_categories.csv").set_index("image_id").scene
    s1_run = sorted(Path("experiments").glob("selector_S1_conservative_*_per_image.csv"))[-1]
    refs = load_references(ids, Path(split_csv), Path("outputs/selector_dev/S1_conservative/masks"), Path("outputs/reference_reviewed_dev/masks"),
                           s1_run, Path("data/review/dev_manual_selection_manifest.csv"), Path("outputs/sam_candidates"))
    images = {i: cv2.imdecode(np.fromfile(str(Path(data) / sp.loc[i, "path"]), np.uint8), cv2.IMREAD_COLOR) for i in ids}
    return {"ids": ids, "images": images, "refs": refs, "expected": {i: int(counts.loc[i, "expected_egg_count"]) for i in ids},
            "tag": {i: ("" if pd.isna(counts.loc[i, "difficulty_tag"]) else counts.loc[i, "difficulty_tag"]) for i in ids},
            "scene": {i: scenes[i] for i in ids}, "s1_run": str(s1_run), "subsets": subsets(ids, "data/review/dev_manual_selection_manifest.csv")}


def code_version() -> str:
    h = hashlib.sha256()
    for f in ("src/segmentation/classical.py", "src/experiments/classical_experiments.py", "src/evaluation/metrics.py"):
        h.update(Path(f).read_bytes())
    return "srchash:" + h.hexdigest()[:12]


def append_log(row: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def observed_failures(df: pd.DataFrame) -> str:
    """Automatic failure summary for the log (qualitative inspection of overlays is recorded separately)."""
    sc = df.groupby("scene").apply(lambda g: pd.Series({"f1": prf(int(g.rev_inst_tp.sum()), int(g.rev_inst_fp.sum()), int(g.rev_inst_fn.sum()))["f1"]}),
                                   include_groups=False).f1.sort_values()
    worst = ", ".join(f"{s}={v:.2f}" for s, v in sc.head(3).items())
    return (f"worst scenes by instance F1 vs R_rev: {worst}; images with empty prediction but non-empty R_rev: "
            f"{int((~df.rev_ref_empty & df.rev_pred_empty).sum())}; images with prediction where R_rev is empty: {int((df.rev_ref_empty & ~df.rev_pred_empty).sum())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/classical")
    ap.add_argument("--skip-loso", action="store_true")
    a = ap.parse_args()
    if LOG.exists():
        raise SystemExit(f"{LOG} exists: the pre-registered experiments run exactly once")
    inputs = load_inputs()
    cache: dict[str, pd.DataFrame] = {}

    def eval_fn(cfg: ClassicalConfig) -> pd.DataFrame:
        if cfg.sha() not in cache:
            cache[cfg.sha()] = evaluate_config(cfg, inputs)
        return cache[cfg.sha()]

    ids = set(inputs["ids"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    cfg, steps = run_sequence(eval_fn, ids)
    nxt = {steps[i]["id"]: steps[i + 1]["id"] for i in range(len(steps) - 1)}
    for s in steps:
        c = s["cfg"]
        df = evaluate_config(c, inputs, Path(a.out) / s["id"])  # re-run to save masks and per-image results of this experiment
        df.to_csv(f"experiments/classical_{s['id']}_per_image.csv", index=False)
        met = {k: aggregate(df, v) for k, v in inputs["subsets"].items()}
        d = s["deltas"]
        append_log({"experiment_id": s["id"], "timestamp": stamp, "code_version": code_version(),
                    "dataset_version": f"split v2_pilot_exposure_controlled; dev ids sha {hashlib.sha256(Path('experiments/classical_dev_image_ids.txt').read_bytes()).hexdigest()[:12]}",
                    "split_seed": 42, "preprocessing": f"blur_sigma={c.blur_sigma}", "color_space": c.colour, "threshold_method": c.threshold,
                    "parameter_values": json.dumps(c.as_dict()), "morphology": f"open {c.open_k}, close {c.close_k}, fill holes, split={c.split_watershed}",
                    "component_filters": f"area {c.min_area_frac}-{c.max_area_frac}; solidity>={c.min_solidity}; aspect<={c.max_aspect}; border_min_area={c.border_min_area_frac}; ellipse_iou>={c.min_ellipse_iou}",
                    "motivation": next(m for e, _, m, _ in EXPERIMENTS if e == s["id"]), "development_metrics": json.dumps(met, default=float),
                    "runtime": json.dumps({"median_s_per_image": met["S_all"]["runtime_median_s"], "total_s": met["S_all"]["runtime_total_s"]}),
                    "observed_failures": observed_failures(df),
                    "decision": s["decision"] + ("" if d is None else f" (deltas vs comparison base: {json.dumps({k: round(v, 4) for k, v in d.items()})})"),
                    "next_change": nxt.get(s["id"], "final configuration")})
        print(s["id"], s["decision"], {k: round(v, 3) for k, v in s["scores"].items()})
    Path("configs").mkdir(exist_ok=True)
    final = {"config": cfg.as_dict(), "config_sha": cfg.sha(), "selected_on": "development images only (88), amendment A6 rule",
             "kept_steps": [s["id"] for s in steps if s["decision"] in ("kept", "baseline")], "code_version": code_version()}
    Path("configs/classical_final_dev.json").write_text(json.dumps(final, indent=1))
    print("FINAL", cfg.sha(), final["kept_steps"])
    if not a.skip_loso:
        lo = loso(eval_fn, inputs)
        lo.to_csv("experiments/classical_loso.csv", index=False)
        for name in ("raw", "rev"):
            d = lo[lo.reference == name]
            p = prf(int(d.inst_tp.sum()), int(d.inst_fp.sum()), int(d.inst_fn.sum()))
            print(f"LOSO pooled held-out instance P/R/F1 vs {name}: {p}")


if __name__ == "__main__":
    main()
