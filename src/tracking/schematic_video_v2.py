"""Footage-free schematic video for the improved tracker (T3b). Draws only what is in tracks.csv: no frame of the source video is read.

Box styles: solid = observed whole egg; dotted = observed fragment of an egg (partly hidden by the hand); dashed + hollow = PREDICTED (not observed);
grey dotted = unassigned fragment (not a track). A red '?' marks an association the tracker itself calls uncertain. Boxes update at 4 Hz and are held.
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
PX0, PY0, PW, PH = 60, 540, 1160, 125
EVENTS = [(480, 558, "annotation: Q lifted by the hand and carried right"), (558, 648, "annotation: Q spinning at right"),
          (648, 750, "annotation: Q carried back to the far left"), (744, 816, "annotation: hand picks up P and carries it right"), (816, 1000, "annotation: P spinning at right")]


def map_box(b):
    return [int(40 + b[0] * 0.4), int(70 + b[1] * 0.4), int(40 + b[2] * 0.4), int(70 + b[3] * 0.4)]


def dashed_rect(img, p1, p2, col, th=2, dash=7, gap=5):
    x1, y1 = p1
    x2, y2 = p2
    for (a, b, c, d) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        n = max(int(np.hypot(c - a, d - b)), 1)
        for k in range(0, n, dash + gap):
            e = min(k + dash, n)
            cv2.line(img, (int(a + (c - a) * k / n), int(b + (d - b) * k / n)), (int(a + (c - a) * e / n), int(b + (d - b) * e / n)), col, th)


def main(tracks_csv: str, out: str, label: str) -> None:
    rows = list(csv.DictReader(open(tracks_csv)))
    by = defaultdict(list)
    for r in rows:
        r["frame"] = int(r["frame"])
        if r["x1"] != "":
            r["box"] = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
            r["cx"] = (r["box"][0] + r["box"][2]) / 2
        by[r["frame"]].append(r)
    obs_hist = defaultdict(list)      # id -> [(frame, cx)] observed (complete or fragment)
    pred_pts = defaultdict(list)
    unc_pts = []
    for r in rows:
        if r["id"] != "" and r["state"] == "observed":
            obs_hist[int(r["id"])].append((r["frame"], r["cx"]))
            if r["uncertain"] == "1":
                unc_pts.append((r["frame"], r["cx"]))
        if r["state"] == "predicted":
            pred_pts[int(r["id"])].append((r["frame"], r["cx"]))
    first = {i: pts[0][0] for i, pts in obs_hist.items()}
    sx = PW / (EXPERIMENT_END - EXPERIMENT_START)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H))
    last_obs = {}
    for fi in range(EXPERIMENT_START, EXPERIMENT_END):
        key = EXPERIMENT_START + ((fi - EXPERIMENT_START) // STEP) * STEP
        f = np.full((H, W, 3), 245, np.uint8)
        cv2.rectangle(f, (40, 70), (808, 502), (200, 205, 215), -1)
        cv2.putText(f, f"{label} | source frame {fi} | boxes update at 4 Hz (sampled frames) and are held", (60, 28), 0, .58, (30, 30, 30), 2)
        cv2.putText(f, "Schematic table = whole 1920x1080 frame at uniform 0.4 scale (no source footage)", (50, 62), 0, .5, (60, 60, 60), 1)
        cur = by.get(key, [])
        shown = set()
        reasons = []
        for r in cur:
            if r["x1"] == "":
                continue
            bb = map_box(r["box"])
            tid = int(r["id"]) if r["id"] != "" else None
            col = COLORS[(tid - 1) % 5] if tid else (130, 130, 130)
            if r["state"] == "observed" and r["kind"] == "complete":
                cv2.rectangle(f, tuple(bb[:2]), tuple(bb[2:]), col, 3)
                txt = f"ID {tid}"
            elif r["state"] == "observed":
                dashed_rect(f, tuple(bb[:2]), tuple(bb[2:]), col, 2, 2, 3)
                txt = f"ID {tid} fragment"
            elif r["state"] == "predicted":
                dashed_rect(f, tuple(bb[:2]), tuple(bb[2:]), col, 2, 10, 6)
                txt = f"ID {tid} PREDICTED (unseen {r['reason'].split(',')[1].strip().split()[0]} sample(s))"
            else:
                dashed_rect(f, tuple(bb[:2]), tuple(bb[2:]), col, 1, 2, 3)
                txt = "unassigned fragment"
            if tid and r["state"] == "observed":
                shown.add(tid); last_obs[tid] = key
            if r["uncertain"] == "1" and r["state"] == "observed":
                txt += "  ?"
                reasons.append(f"ID {tid}: {r['reason']}")
            cv2.putText(f, txt, (bb[0], max(bb[1] - 6, 82)), 0, .55, (0, 0, 200) if "?" in txt else col, 2)
        # identity status panel: identities are never silently deleted
        y = 440
        for tid in sorted(first):
            if first[tid] > key:
                continue
            col = COLORS[(tid - 1) % 5]
            if tid in shown:
                st = "observed"
            elif any(r["state"] == "predicted" and r["id"] == str(tid) for r in cur):
                st = "not observed now: position PREDICTED"
            else:
                st = f"unseen since frame {last_obs.get(tid, key)}; identity kept, position unknown"
            cv2.putText(f, f"ID {tid}: {st}", (830, 100 + 26 * (tid - 1)), 0, .5, col, 2)
        # trajectory plot
        cv2.rectangle(f, (PX0, PY0), (PX0 + PW, PY0 + PH), (255, 255, 255), -1)
        for ident, a, b, why in GAPS:
            cv2.rectangle(f, (int(PX0 + (a - EXPERIMENT_START) * sx), PY0), (int(PX0 + (b - EXPERIMENT_START) * sx), PY0 + PH), (225, 225, 225), -1)
            cv2.putText(f, f"{ident} hidden (annot.)", (int(PX0 + (a - EXPERIMENT_START) * sx) + 4, PY0 + 14), 0, .38, (90, 90, 90), 1)

        def pt(fr, cx):
            return int(PX0 + (fr - EXPERIMENT_START) * sx), int(PY0 + PH - cx / 1920 * PH)

        for tid, pts in obs_hist.items():
            col = COLORS[(tid - 1) % 5]
            pp = [(fr, cx) for fr, cx in pts if fr <= key]
            for (f0, x0), (f1, x1) in zip(pp, pp[1:]):
                a, b = pt(f0, x0), pt(f1, x1)
                if f1 - f0 <= STEP:
                    cv2.line(f, a, b, col, 2)
                else:
                    n = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) // 8), 1)
                    for k in range(0, n, 2):
                        cv2.line(f, (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n)), (int(a[0] + (b[0] - a[0]) * min(k + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(k + 1, n) / n)), col, 1)
        for tid, pts in pred_pts.items():
            for fr, cx in pts:
                if fr <= key:
                    cv2.circle(f, pt(fr, cx), 3, COLORS[(tid - 1) % 5], 1)
        for fr, cx in unc_pts:
            if fr <= key:
                x, yy = pt(fr, cx)
                cv2.drawMarker(f, (x, yy), (0, 0, 220), cv2.MARKER_DIAMOND, 12, 2)
        cv2.line(f, (int(PX0 + (fi - EXPERIMENT_START) * sx), PY0), (int(PX0 + (fi - EXPERIMENT_START) * sx), PY0 + PH), (0, 0, 0), 1)
        cv2.rectangle(f, (PX0, PY0), (PX0 + PW, PY0 + PH), (120, 120, 120), 1)
        cv2.putText(f, "x position vs time: solid = consecutive observations, dashed = unobserved gap (interpolation), hollow circle = predicted, red diamond = uncertain association; black = now", (PX0, PY0 - 8), 0, .4, (60, 60, 60), 1)
        ev = [t for a, b, t in EVENTS if a <= fi < b]
        cv2.putText(f, ev[-1] if ev else "annotation: both eggs visible", (60, 692), 0, .55, (40, 40, 40), 1)
        cv2.putText(f, ("uncertain: " + "; ".join(reasons)) if reasons else "no uncertain associations at this sample", (60, 712), 0, .5, (0, 0, 190) if reasons else (90, 90, 90), 1)
        vw.write(f)
    vw.release()
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
