"""Detect + track on the sampled frames of the experiment interval; save per-frame CSV and overlays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .detect import VideoDetectorConfig, detect
from .sampling import is_dev, sample_frames
from .tracker import Tracker, TrackerConfig

VIDEO = "scratch/video_dl/egg_video.f399.mp4"
COLORS = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0), (0, 0, 255), (255, 0, 0)]


def get_detections(video, frames, dcfg, cache="outputs/video/detections_cache"):
    """Detections + 480x270 thumbnails per sampled frame; cached by (video size, detector sha, frame list)."""
    key = hashlib.sha256(f"{Path(video).stat().st_size}|{dcfg.sha()}|{frames[0]}-{frames[-1]}-{len(frames)}".encode()).hexdigest()[:12]
    jp, zp = Path(cache) / f"{key}.json", Path(cache) / f"{key}.npz"
    if jp.exists() and zp.exists():
        meta = json.loads(jp.read_text())
        return meta["dets"], dict(zip(frames, np.load(zp)["thumbs"])), meta["detect_s"]
    Path(cache).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    all_d, thumbs, det_t = [], [], []
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {fi}")
        t0 = time.perf_counter()
        d = detect(f, dcfg)
        det_t.append(time.perf_counter() - t0)
        all_d.append([{k: v for k, v in e.items() if not k.startswith("_")} for e in d])
        thumbs.append(cv2.resize(f, (480, 270)))
    np.savez_compressed(zp, thumbs=np.stack(thumbs))
    jp.write_text(json.dumps({"dets": all_d, "detect_s": det_t, "frames": frames}))
    return all_d, dict(zip(frames, thumbs)), det_t


def run(video: str = VIDEO, out: str = "outputs/video/tracking", dcfg=None, tcfg=None, frames=None, overlays=True):
    dcfg, tcfg = dcfg or VideoDetectorConfig(), tcfg or TrackerConfig()
    frames = frames or sample_frames()
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_d, thumbs, det_t = get_detections(video, frames, dcfg)
    tr = Tracker(tcfg)
    rows, shots = [], {}
    t_track = []
    for step, fi in enumerate(frames):
        dets = all_d[step]
        t0 = time.perf_counter()
        res = tr.update(step, dets)
        t_track.append(time.perf_counter() - t0)
        sp = "dev" if is_dev(fi) else "heldout"
        for r in res:
            d = dets[r["det"]]
            rows.append({"frame": fi, "step": step, "split": sp, "id": r["id"], "new": int(r["new"]),
                         "x1": round(d["bbox"][0], 1), "y1": round(d["bbox"][1], 1),
                         "x2": round(d["bbox"][2], 1), "y2": round(d["bbox"][3], 1)})
        if not res:
            rows.append({"frame": fi, "step": step, "split": sp, "id": "", "new": 0, "x1": "", "y1": "", "x2": "", "y2": ""})
        if overlays:
            s = thumbs[fi].copy(); k = 480 / 1920
            for r in res:
                b = [int(v * k) for v in dets[r["det"]]["bbox"]]
                col = COLORS[(r["id"] - 1) % len(COLORS)]
                cv2.rectangle(s, b[:2], b[2:], col, 2); cv2.putText(s, f"id{r['id']}", (b[0], b[1] - 4), 0, .6, col, 2)
            cv2.putText(s, f"{fi}{'' if is_dev(fi) else ' H'}", (5, 20), 0, .6, (0, 255, 255), 2)
            shots[fi] = s
    fields = ["frame", "step", "split", "id", "new", "x1", "y1", "x2", "y2"]
    with open(out_dir / "tracks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    meta = {"detector": dcfg.sha(), "tracker": tcfg.sha(), "frames": len(frames),
            "median_detect_ms_960w": round(1000 * float(np.median(det_t)), 1),
            "median_track_ms": round(1000 * float(np.median(t_track)), 3),
            "tracks_csv_sha": hashlib.sha256((out_dir / "tracks.csv").read_bytes()).hexdigest()[:16]}
    return rows, tr, shots, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/video/tracking")
    a = ap.parse_args()
    rows, tr, shots, meta = run(out=a.out)
    print(meta, [(t.tid, t.first_step, t.last_step, t.hits) for t in tr.all_tracks()])
    fr = sorted(shots)
    for n, k in enumerate(range(0, len(fr), 24)):
        tiles = [shots[x] for x in fr[k:k + 24]]
        while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(f"{a.out}/dense_{n}.jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [1, 80])
