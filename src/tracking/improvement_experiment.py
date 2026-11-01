"""Tracking-improvement variants (T2, T3) evaluated against the unchanged T0/T1 records. See experiments/tracking_improvement_audit.md.

Scoring uses ONLY complete, observed detections (exactly what T0/T1 were scored on); partial detections and predicted boxes are
reported separately. The identity annotation is not changed. Every run appends one row per subset to
experiments/video_tracking_improvement_log.csv (append-only, failures included).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .detect import VideoDetectorConfig
from .detect_v2 import PartialConfig, detect_with_partials
from .evaluate import evaluate, load_segments
from .sampling import is_dev, sample_frames, split_frames
from .tracker_v2 import KalmanTracker, KConfig

CACHE = Path("scratch/framecache/frames960.npz")
VIDEO = "scratch/video_dl/egg_video.f399.mp4"
LOG = Path("experiments/video_tracking_improvement_log.csv")
FIELDS = ["variant", "detector_sha", "partial_sha", "tracker_sha", "tracker_params", "subset", "frames_used", "expected_eggs", "complete_detections", "coverage",
          "missed", "extra", "id_switches", "distinct_ids", "extra_ids", "gaps_retained", "partial_assigned", "partial_unassigned", "predicted_rows",
          "uncertain_rows", "median_detect_ms_960w", "median_track_ms", "note"]
VARIANTS = {
    "T2_kalman": (PartialConfig(enabled=False), KConfig()),
    "T3_kalman_partial": (PartialConfig(enabled=True), KConfig()),
    # T3b: added AFTER T3 failed the pre-registered 'no implausible predicted boxes' criterion (predicted centre left the frame at frames 780-792);
    # the only changes are the bounded prediction and a 3-step display budget. Post-result variant, disclosed as such.
    "T3b_kalman_partial_bounded": (PartialConfig(enabled=True), KConfig(frame_w=1920, frame_h=1080, max_predicted_display=3)),
}


def ensure_cache() -> None:
    if CACHE.exists():
        return
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    fr = set(sample_frames()) | set(range(690, 790, 2))
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 240)
    out = {}
    for i in range(240, 1000):
        if i in fr:
            ok, f = cap.read()
            out[str(i)] = cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA)
        else:
            cap.grab()
    np.savez_compressed(CACHE, **out)


def to_original(d: dict) -> dict:
    """The cache is the 960 px working frame, so detector coordinates are in 960 space: scale to the 1920 px original."""
    e = dict(d)
    e["bbox"] = [v * 2 for v in d["bbox"]]
    e["centroid"] = [v * 2 for v in d["centroid"]]
    e["area"] = d["area"] * 4
    return e


def run_variant(name: str, part: PartialConfig, kcfg: KConfig, out_root="outputs/video"):
    ensure_cache()
    Z = np.load(CACHE)
    frames = sample_frames()
    base = VideoDetectorConfig()
    tr = KalmanTracker(kcfg)
    rows, det_t, trk_t = [], [], []
    for step, fi in enumerate(frames):
        img = Z[str(fi)]
        t0 = time.perf_counter()
        dets = [to_original(d) for d in detect_with_partials(img, base, part)]
        det_t.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        obs, pred = tr.update(step, dets)
        trk_t.append(time.perf_counter() - t1)
        sp = "dev" if is_dev(fi) else "heldout"
        for o in obs:
            d = dets[o["det"]]
            rows.append({"frame": fi, "step": step, "split": sp, "id": "" if o["id"] is None else o["id"], "kind": d["kind"],
                         "state": "observed" if o["id"] is not None else "unassigned", "uncertain": int(o["uncertain"]), "reason": o["reason"],
                         "new": int(o["new"]), "x1": round(d["bbox"][0], 1), "y1": round(d["bbox"][1], 1), "x2": round(d["bbox"][2], 1), "y2": round(d["bbox"][3], 1)})
        for p in pred:
            rows.append({"frame": fi, "step": step, "split": sp, "id": p["id"], "kind": "predicted", "state": "predicted", "uncertain": 1,
                         "reason": f"predicted, {p['age']} steps since last observation", "new": 0,
                         "x1": round(p["bbox"][0], 1), "y1": round(p["bbox"][1], 1), "x2": round(p["bbox"][2], 1), "y2": round(p["bbox"][3], 1)})
        if not obs and not pred:
            rows.append({"frame": fi, "step": step, "split": sp, "id": "", "kind": "", "state": "none", "uncertain": 0, "reason": "", "new": 0, "x1": "", "y1": "", "x2": "", "y2": ""})
    out = Path(out_root) / f"tracking_{name}"
    out.mkdir(parents=True, exist_ok=True)
    fields = ["frame", "step", "split", "id", "kind", "state", "uncertain", "reason", "new", "x1", "y1", "x2", "y2"]
    with open(out / "tracks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return rows, tr, float(np.median(det_t) * 1000), float(np.median(trk_t) * 1000)


def score(rows, frames, segs, amb, subset):
    ev_rows = [r for r in rows if r["state"] == "observed" and r["kind"] == "complete" and r["id"] != ""]
    return evaluate(ev_rows, frames, segs, amb, subset)


def main(names=None) -> None:
    segs = load_segments()
    amb = {int(x) for x in Path("data/video/ambiguous_frames.txt").read_text().split()}
    frames = sample_frames()
    dev, held = split_frames(frames)
    base = VideoDetectorConfig()
    for name, (part, kcfg) in VARIANTS.items():
        if names and name not in names:
            continue
        rows, tr, det_ms, trk_ms = run_variant(name, part, kcfg)
        for sub, fs in (("dev", set(dev)), ("heldout", set(held)), ("all", set(frames))):
            r = score(rows, frames, segs, amb, fs)
            sr = [x for x in rows if x["frame"] in fs]
            gr = ["%s:%s" % (g["identity"], "n/a" if g["retained"] is None else int(g["retained"])) for g in r["gaps"]]
            new = not LOG.exists()
            with open(LOG, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow({"variant": name, "detector_sha": base.sha(), "partial_sha": part.sha(), "tracker_sha": kcfg.sha(), "tracker_params": json.dumps(kcfg.__dict__),
                            "subset": sub, "frames_used": r["frames_used"], "expected_eggs": r["expected_eggs"], "complete_detections": r["detections"], "coverage": round(r["coverage"], 4),
                            "missed": r["missed"], "extra": r["extra"], "id_switches": r["id_switches"], "distinct_ids": len(r["distinct_ids"]), "extra_ids": r["extra_ids"],
                            "gaps_retained": " ".join(gr), "partial_assigned": sum(1 for x in sr if x["kind"] == "partial" and x["state"] == "observed"),
                            "partial_unassigned": sum(1 for x in sr if x["state"] == "unassigned"), "predicted_rows": sum(1 for x in sr if x["state"] == "predicted"),
                            "uncertain_rows": sum(int(x["uncertain"]) for x in sr if x["state"] == "observed"), "median_detect_ms_960w": round(det_ms, 2),
                            "median_track_ms": round(trk_ms, 3), "note": ""})
            print(name, sub, {k: r[k] for k in ("frames_used", "expected_eggs", "detections", "coverage", "missed", "extra", "id_switches", "ids_per_identity", "distinct_ids", "track_purity")})
            print("   gaps:", [(g["identity"], g["start"], g["end"], g["retained"], g.get("id_before"), g.get("id_after")) for g in r["gaps"]])


def sensitivity(names=None) -> None:
    """One-factor-at-a-time robustness check of T2/T3 (post-hoc, NOT used for selection). One log row per run ('all' subset)."""
    from dataclasses import replace
    segs = load_segments()
    amb = {int(x) for x in Path("data/video/ambiguous_frames.txt").read_text().split()}
    frames = sample_frames()
    grid = {"q_acc": [0.1, 0.2, 0.35, 0.7, 1.4], "sigma_meas": [0.04, 0.08, 0.16], "vel_damp": [0.5, 0.85, 1.0], "chi2_gate": [9.0, 16.0, 25.0], "iou_weight": [0.0, 0.3, 1.0]}
    base = VideoDetectorConfig()
    for vname in (names or ("T2_kalman", "T3_kalman_partial")):
        part, k0 = VARIANTS[vname]
        for key, vals in grid.items():
            for v in vals:
                kc = replace(k0, **{key: v})
                rows, tr, det_ms, trk_ms = run_variant(f"{vname}@{key}={v}", part, kc, out_root="scratch/sens")
                r = score(rows, frames, segs, amb, set(frames))
                gr = ["%s:%s" % (g["identity"], "n/a" if g["retained"] is None else int(g["retained"])) for g in r["gaps"]]
                with open(LOG, "a", newline="") as fh:
                    csv.DictWriter(fh, fieldnames=FIELDS).writerow({"variant": f"{vname}@{key}={v}", "detector_sha": base.sha(), "partial_sha": part.sha(), "tracker_sha": kc.sha(),
                        "tracker_params": json.dumps(kc.__dict__), "subset": "all", "frames_used": r["frames_used"], "expected_eggs": r["expected_eggs"], "complete_detections": r["detections"],
                        "coverage": round(r["coverage"], 4), "missed": r["missed"], "extra": r["extra"], "id_switches": r["id_switches"], "distinct_ids": len(r["distinct_ids"]),
                        "extra_ids": r["extra_ids"], "gaps_retained": " ".join(gr), "partial_assigned": sum(1 for x in rows if x["kind"] == "partial" and x["state"] == "observed"),
                        "partial_unassigned": sum(1 for x in rows if x["state"] == "unassigned"), "predicted_rows": sum(1 for x in rows if x["state"] == "predicted"),
                        "uncertain_rows": sum(int(x["uncertain"]) for x in rows if x["state"] == "observed"), "median_detect_ms_960w": round(det_ms, 2), "median_track_ms": round(trk_ms, 3),
                        "note": "sensitivity (post-hoc robustness check, not used for selection)"})
                print(f"{vname:18s} {key}={v:<5} idsw={r['id_switches']} ids={len(r['distinct_ids'])} gaps={' '.join(gr)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--sensitivity"]:
        sensitivity(sys.argv[2:] or None)
    else:
        main(sys.argv[1:] or None)
