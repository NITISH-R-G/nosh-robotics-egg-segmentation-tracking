"""Run the two pre-declared tracker variants and write the proxy evaluation (dev and held-out reported separately)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .detect import VideoDetectorConfig
from .evaluate import evaluate, load_segments
from .run_video import run
from .sampling import is_dev, sample_frames, split_frames
from .tracker import TrackerConfig

VARIANTS = {"T0_gated": TrackerConfig(), "T1_gated_plus_two_egg_prior": TrackerConfig(max_tracks=2)}
LOG = Path("experiments/video_tracking_log.csv")
FIELDS = ["variant", "detector_sha", "tracker_sha", "tracker_params", "subset", "frames_used", "expected_eggs", "detections",
          "coverage", "missed", "extra", "id_switches", "distinct_ids", "extra_ids", "gaps_retained", "median_detect_ms_960w", "median_track_ms"]


def main():
    segs = load_segments()
    amb = {int(x) for x in Path("data/video/ambiguous_frames.txt").read_text().split()}
    frames = sample_frames()
    dev, held = split_frames(frames)
    dcfg = VideoDetectorConfig()
    results = {}
    for name, tcfg in VARIANTS.items():
        out = f"outputs/video/tracking_{name}"
        rows, tr, shots, meta = run(out=out, dcfg=dcfg, tcfg=tcfg)
        for sub, fs in (("dev", set(dev)), ("heldout", set(held)), ("all", set(frames))):
            r = evaluate(rows, frames, segs, amb, fs)
            results[(name, sub)] = r
            gr = [("%s:%s" % (g["identity"], "n/a" if g["retained"] is None else int(g["retained"]))) for g in r["gaps"]]
            new = not LOG.exists()
            with open(LOG, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow({"variant": name, "detector_sha": meta["detector"], "tracker_sha": meta["tracker"],
                            "tracker_params": json.dumps(tcfg.__dict__), "subset": sub, "frames_used": r["frames_used"],
                            "expected_eggs": r["expected_eggs"], "detections": r["detections"], "coverage": round(r["coverage"], 4),
                            "missed": r["missed"], "extra": r["extra"], "id_switches": r["id_switches"],
                            "distinct_ids": len(r["distinct_ids"]), "extra_ids": r["extra_ids"], "gaps_retained": " ".join(gr),
                            "median_detect_ms_960w": meta["median_detect_ms_960w"], "median_track_ms": meta["median_track_ms"]})
        fr = sorted(shots)
        for n, k in enumerate(range(0, len(fr), 24)):
            tiles = [shots[x] for x in fr[k:k + 24]]
            while len(tiles) % 4:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(f"{out}/dense_{n}.jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [1, 80])
    (Path("outputs/video") / "evaluation.json").write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in results.items()}, indent=1, default=str))
    for (name, sub), r in results.items():
        print(name, sub, {k: r[k] for k in ("frames_used", "expected_eggs", "detections", "coverage", "missed", "extra", "id_switches", "ids_per_identity", "distinct_ids", "track_purity")})
        print("   gaps:", [(g["identity"], g["start"], g["end"], g["retained"], g.get("id_before"), g.get("id_after")) for g in r["gaps"]])


if __name__ == "__main__":
    main()
