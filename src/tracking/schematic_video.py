"""Original-footage-free demonstration video: tracker output drawn on a blank schematic table plus a growing trajectory plot.

Only the saved tracks.csv boxes and the proxy identity annotation are used; no frame of the third-party video is read or shown.
Boxes update at 4 Hz (sampled frames) and are held between updates, as in the footage-based export.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import cv2
import numpy as np

from .evaluate import GAPS
from .sampling import EXPERIMENT_END, EXPERIMENT_START, STEP

COLORS = [(255, 114, 0), (0, 158, 230), (167, 121, 204), (115, 158, 0), (0, 94, 213)]  # BGR
W, H = 1280, 720
EVENTS = [(480, 558, "Q lifted and carried right by the hand (hidden)"), (558, 648, "Q spinning at right"), (648, 750, "Q carried back to the far left (hidden)"),
          (744, 750, "hand picks up P next to Q (both near the hand); no detection"),
          (750, 816, "SWITCH: ID 2 (was P) now sits on stationary Q; P is carried by the hand, undetected"), (816, 1000, "P spinning at right (new ID)")]


def main(tracks_csv: str, out: str, label: str) -> None:
    by = defaultdict(list)
    hist = defaultdict(list)
    for r in csv.DictReader(open(tracks_csv)):
        if r["id"] != "":
            cx = (float(r["x1"]) + float(r["x2"])) / 2
            by[int(r["frame"])].append((int(r["id"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
            hist[int(r["id"])].append((int(r["frame"]), cx))
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H))
    px0, py0, pw, ph = 60, 450, 1160, 215       # plot area
    sx = pw / (EXPERIMENT_END - EXPERIMENT_START)
    for fi in range(EXPERIMENT_START, EXPERIMENT_END):
        key = EXPERIMENT_START + ((fi - EXPERIMENT_START) // STEP) * STEP
        f = np.full((H, W, 3), 245, np.uint8)
        cv2.rectangle(f, (40, 70), (1240, 415), (200, 205, 215), -1)           # schematic table, not footage
        cv2.putText(f, "Schematic table (no source footage): boxes = tracker output at uniform 0.625 scale, cropped to the egg band (source y 365-917)", (50, 62), 0, .55, (60, 60, 60), 1)
        for tid, b in by.get(key, []):
            col = COLORS[(tid - 1) % 5]
            bb = [int(40 + b[0] * 0.625), int(70 + (b[1] - 365) * 0.625), int(40 + b[2] * 0.625), int(70 + (b[3] - 365) * 0.625)]   # uniform scale 0.625, vertical crop to source y 365-917
            cv2.rectangle(f, bb[:2], bb[2:], col, 3)
            cv2.putText(f, f"ID {tid}", (bb[0], bb[1] - 6), 0, .7, col, 2)
        # trajectory plot (x centre vs frame) drawn up to the current sampled frame
        cv2.rectangle(f, (px0, py0), (px0 + pw, py0 + ph), (255, 255, 255), -1)
        for ident, a, b, why in GAPS:
            cv2.rectangle(f, (int(px0 + (a - EXPERIMENT_START) * sx), py0), (int(px0 + (b - EXPERIMENT_START) * sx), py0 + ph), (225, 225, 225), -1)
            cv2.putText(f, f"{ident} hidden", (int(px0 + (a - EXPERIMENT_START) * sx) + 4, py0 + 14), 0, .4, (90, 90, 90), 1)
        for tid, pts in hist.items():
            col = COLORS[(tid - 1) % 5]
            pp = [(int(px0 + (fr - EXPERIMENT_START) * sx), int(py0 + ph - cx / 1920 * ph)) for fr, cx in pts if fr <= key]
            fr = [fr for fr, cx in pts if fr <= key]
            for (a, b), (f0, f1) in zip(zip(pp, pp[1:]), zip(fr, fr[1:])):
                if f1 - f0 <= STEP:
                    cv2.line(f, a, b, col, 2)
                else:   # no detection for >= 1 sample: dashed straight line is interpolation, not observed motion
                    n = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) // 8), 1)
                    for k in range(0, n, 2):
                        cv2.line(f, (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n)), (int(a[0] + (b[0] - a[0]) * min(k + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(k + 1, n) / n)), col, 2)
        cv2.line(f, (int(px0 + (fi - EXPERIMENT_START) * sx), py0), (int(px0 + (fi - EXPERIMENT_START) * sx), py0 + ph), (0, 0, 0), 1)
        cv2.rectangle(f, (px0, py0), (px0 + pw, py0 + ph), (120, 120, 120), 1)
        ev = [t for a, b, t in EVENTS if a <= fi < b]
        cv2.putText(f, "Event (proxy annotation): " + (ev[-1] if ev else "both eggs visible and stationary / spinning"), (60, 702), 0, .6, (0, 0, 160) if "wrong" in " ".join(ev) else (40, 40, 40), 2)
        cv2.putText(f, f"{label} | source frame {fi} | boxes updated at 4 Hz, held between updates", (60, 30), 0, .6, (30, 30, 30), 2)
        cv2.putText(f, "x position of egg-box centres over time; solid = detected at consecutive samples, dashed = no detection in between (straight line is interpolation); black line = current frame", (px0, py0 - 8), 0, .42, (60, 60, 60), 1)
        if not by.get(key):
            cv2.putText(f, "no egg detected at this 4 Hz sample", (50, 408), 0, .6, (0, 0, 160), 2)
        vw.write(f)
    vw.release()
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
