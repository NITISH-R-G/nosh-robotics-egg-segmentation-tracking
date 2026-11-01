"""Read-only audit of the exported annotated video and the tracking result files (writes experiments/video_artifact_audit.json)."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.tracking.evaluate import evaluate, load_segments
from src.tracking.sampling import EXPERIMENT_END, EXPERIMENT_START, STEP, sample_frames
from src.tracking.run_video import COLORS

VID = "outputs/video/final/egg_tracking_T1.mp4"


def main() -> None:
    out = {}
    cap = cv2.VideoCapture(VID)
    out["opened"] = cap.isOpened()
    out["header"] = {"frames": int(cap.get(7)), "fps": cap.get(5), "w": int(cap.get(3)), "h": int(cap.get(4))}
    rows = list(csv.DictReader(open("outputs/video/tracking_T1_gated_plus_two_egg_prior/tracks.csv")))
    by = defaultdict(list)
    for r in rows:
        if r["id"] != "":
            by[int(r["frame"])].append((int(r["id"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
    n = 0
    cent = {}   # id -> list of (frame_index, centroid of colour mask)
    k = 1280 / 1920
    bad_match = 0
    checked = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fi = EXPERIMENT_START + n
        n += 1
        key = EXPERIMENT_START + ((fi - EXPERIMENT_START) // STEP) * STEP
        for tid, b in by.get(key, []):
            col = np.array(COLORS[(tid - 1) % len(COLORS)], int)
            bb = [int(v * k) for v in b]
            # sample the top edge of the expected box: colour must dominate there (mp4v compression -> tolerance)
            seg = f[bb[1] + 1: bb[1] + 3, bb[0] + 5: bb[2] - 5].reshape(-1, 3).astype(int)
            near = (np.abs(seg - col).sum(1) < 120).mean() if seg.size else 0
            checked += 1
            bad_match += int(near < 0.5)
    out["decoded_frames"] = n
    out["frames_expected_from_interval"] = EXPERIMENT_END - EXPERIMENT_START
    out["box_edge_checks"] = checked
    out["box_edge_checks_failed"] = int(bad_match)
    # update cadence: expected box positions only change at sampled frames
    out["box_source_rule"] = "frame fi shows boxes of sampled frame 240 + ((fi-240)//6)*6 (src/tracking/export_video.py)"
    out["duration_s"] = n / out["header"]["fps"] if out["header"]["fps"] else None
    segs = load_segments()
    amb = {int(x) for x in Path("data/video/ambiguous_frames.txt").read_text().split()}
    frames = sample_frames()
    ev = json.loads(Path("outputs/video/evaluation.json").read_text())
    log = list(csv.DictReader(open("experiments/video_tracking_log.csv")))
    recomputed = {}
    for var in ("T0_gated", "T1_gated_plus_two_egg_prior"):
        rr = list(csv.DictReader(open(f"outputs/video/tracking_{var}/tracks.csv")))
        for sub in ("dev", "heldout", "all"):
            from src.tracking.sampling import is_dev
            fs = {f for f in frames if (is_dev(f) if sub == "dev" else (not is_dev(f) if sub == "heldout" else True))}
            r = evaluate(rr, frames, segs, amb, fs)
            lg = next(x for x in log if x["variant"] == var and x["subset"] == sub)
            recomputed[f"{var}|{sub}"] = {"frames_used": r["frames_used"], "expected": r["expected_eggs"], "detections": r["detections"], "missed": r["missed"], "extra": r["extra"],
                                          "id_switches": r["id_switches"], "distinct_ids": len(r["distinct_ids"]),
                                          "matches_log": (int(lg["frames_used"]), int(lg["expected_eggs"]), int(lg["detections"]), int(lg["missed"]), int(lg["extra"]), int(lg["id_switches"]), int(lg["distinct_ids"])) ==
                                                         (r["frames_used"], r["expected_eggs"], r["detections"], r["missed"], r["extra"], r["id_switches"], len(r["distinct_ids"])),
                                          "matches_evaluation_json": ev[f"{var}|{sub}"]["id_switches"] == r["id_switches"] and ev[f"{var}|{sub}"]["missed"] == r["missed"]}
    out["recomputed"] = recomputed
    Path("experiments/video_artifact_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
