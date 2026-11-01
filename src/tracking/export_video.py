"""Export an annotated video: the sampled-frame (4 Hz) track boxes held for 6 frames over the full-rate source video."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

from .run_video import COLORS, VIDEO
from .sampling import EXPERIMENT_END, EXPERIMENT_START, STEP


def main(tracks_csv: str, out: str, label: str) -> None:
    by = defaultdict(list)
    for r in csv.DictReader(open(tracks_csv)):
        if r["id"] != "":
            by[int(r["frame"])].append((int(r["id"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, EXPERIMENT_START)
    W, H = 1280, 720
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H))
    k = W / 1920
    for fi in range(EXPERIMENT_START, EXPERIMENT_END):
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        key = EXPERIMENT_START + ((fi - EXPERIMENT_START) // STEP) * STEP
        for tid, b in by.get(key, []):
            col = COLORS[(tid - 1) % len(COLORS)]
            bb = [int(v * k) for v in b]
            cv2.rectangle(f, bb[:2], bb[2:], col, 3)
            cv2.putText(f, f"ID {tid}", (bb[0], max(bb[1] - 8, 20)), 0, 0.9, col, 2)
        cv2.putText(f, f"{label} | frame {fi} | boxes updated at 4 Hz, held between updates", (10, H - 12), 0, 0.55, (255, 255, 255), 1)
        vw.write(f)
    vw.release()
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
