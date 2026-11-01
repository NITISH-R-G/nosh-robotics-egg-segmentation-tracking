"""Render the SAVED T3b tracking output over the original footage (LOCAL USE; the footage is third-party and is not to be published).

No tracker or detector is run: the script only reads `outputs/video/tracking_T3b_kalman_partial_bounded/tracks.csv` (127 sampled frames, every 6th source frame, 240..996) and draws it on
`scratch/video_dl/egg_video.f399.mp4`. The tracker updates at 4 Hz; each update's boxes are held for 6 source frames (24 fps), so the overlay never shows more than the saved output.
Row semantics in tracks.csv: state `observed` + kind `complete` (whole-egg detection), `observed` + `partial` (fragment assigned to a track), `predicted` (model extrapolation, NOT an
observation), `unassigned` + `partial` (fragment that no track took). `uncertain`/`reason` are drawn when set. Existing videos are never overwritten.
Usage: python -m src.tracking.render_t3b_overlay [out_path]
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

VIDEO = "scratch/video_dl/egg_video.f399.mp4"
TRACKS = "outputs/video/tracking_T3b_kalman_partial_bounded/tracks.csv"
DEFAULT_OUT = "outputs/video/final/egg_tracking_T3b_footage_overlay_LOCAL_ONLY_v2.mp4"
START, END, STEP = 240, 1000, 6            # source frames rendered: 240..999 (the interval the saved sampling covers; same as the existing T1 export)
W, H = 1280, 720
K = W / 1920
COLORS = {1: (60, 200, 60), 2: (0, 165, 255)}      # BGR: ID 1 green, ID 2 orange (same ids as the T1 video)
GREY, RED, WHITE = (170, 170, 170), (40, 40, 255), (255, 255, 255)


def dashed(img, p1, p2, col, th, dash, gap):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = max(abs(dx), abs(dy))
        if n == 0:
            continue
        t = 0
        while t < n:
            s, e = t, min(t + dash, n)
            cv2.line(img, (int(a[0] + dx * s / n), int(a[1] + dy * s / n)), (int(a[0] + dx * e / n), int(a[1] + dy * e / n)), col, th, cv2.LINE_AA)
            t += dash + gap


def text(img, s, org, col, scale=0.55, th=1):
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, 2), (0, -2), (2, 0), (-2, 0)):      # outline as offset copies at the SAME thickness (a thicker pass lays glyphs out wider and leaves ghost text)
        cv2.putText(img, s, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def load():
    by = defaultdict(list)
    for r in csv.DictReader(open(TRACKS)):
        by[int(r["frame"])].append(r)
    return by


def draw_frame(f, fi, rows, seen):
    key = START + ((fi - START) // STEP) * STEP
    held = fi - key
    status = {}
    for r in rows:
        if r["state"] == "none" or any(r[c] == "" for c in ("x1", "y1", "x2", "y2")):      # a row with no box has nothing to draw
            continue
        box = [int(float(r[c]) * K) for c in ("x1", "y1", "x2", "y2")]
        tid = int(r["id"]) if r["id"] != "" else None
        col = COLORS.get(tid, GREY) if tid is not None else GREY
        unc = r["uncertain"] == "1"
        reason = r["reason"]
        if r["state"] == "observed" and r["kind"] == "complete":
            cv2.rectangle(f, box[:2], box[2:], col, 3)
            lab = f"ID {tid}" + ("  ? identity uncertain: " + reason if unc else "")
            text(f, lab, (box[0], max(box[1] - 8, 40)), RED if unc else col, 0.7, 2)
            status[tid] = "OBSERVED (whole egg)" + (" - identity UNCERTAIN" if unc else "")
        elif r["state"] == "observed" and r["kind"] == "partial":
            dashed(f, box[:2], box[2:], col, 2, 10, 6)
            lab = f"ID {tid} fragment (partial detection)" + ("  ? " + reason if unc and reason else "")
            text(f, lab, (box[0], max(box[1] - 8, 40)), RED if unc else col, 0.55, 1)
            status[tid] = "OBSERVED fragment only" + (" - UNCERTAIN" if unc else "")
        elif r["state"] == "predicted":
            dashed(f, box[:2], box[2:], col, 2, 3, 7)
            text(f, f"ID {tid} PREDICTED, not observed ({reason.split(',')[1].strip() if ',' in reason else reason})", (box[0], max(box[1] - 8, 40)), (0, 255, 255), 0.55, 1)
            status[tid] = "PREDICTED only (no detection)"
        else:                                           # unassigned partial: no track took it
            dashed(f, box[:2], box[2:], GREY, 1, 3, 6)
            text(f, "unassigned fragment (no track)", (box[0], max(box[3] + 18, 40)), GREY, 0.5, 1)
    for tid in sorted(seen | set(status)):
        if tid not in status:
            status[tid] = "NO ESTIMATE shown (not observed, no prediction)"
    # legend
    x0, y0 = 10, 8
    cv2.rectangle(f, (x0, y0), (x0 + 600, y0 + 124), (0, 0, 0), -1)
    cv2.rectangle(f, (x0, y0), (x0 + 600, y0 + 124), (90, 90, 90), 1)
    text(f, "T3b tracker output (Kalman + partial fragments, bounded) - saved results, NOT re-run", (x0 + 6, y0 + 18), WHITE, 0.5, 1)
    cv2.rectangle(f, (x0 + 8, y0 + 28), (x0 + 38, y0 + 42), COLORS[1], 3)
    text(f, "solid = observed whole-egg detection (colour = track ID)", (x0 + 46, y0 + 41), WHITE, 0.45)
    dashed(f, (x0 + 8, y0 + 50), (x0 + 38, y0 + 64), COLORS[1], 2, 10, 6)
    text(f, "dashed = observed fragment (partial detection) assigned to the ID", (x0 + 46, y0 + 63), WHITE, 0.45)
    dashed(f, (x0 + 8, y0 + 72), (x0 + 38, y0 + 86), COLORS[1], 2, 3, 7)
    text(f, "dotted + yellow text = PREDICTED position, not a detection", (x0 + 46, y0 + 85), WHITE, 0.45)
    dashed(f, (x0 + 8, y0 + 94), (x0 + 38, y0 + 108), GREY, 1, 3, 6)
    text(f, "grey dotted = unassigned fragment; red text = identity uncertain", (x0 + 46, y0 + 107), WHITE, 0.45)
    # per-ID status panel
    px, py = W - 400, 8
    cv2.rectangle(f, (px, py), (W - 10, py + 22 + 20 * max(len(status), 1)), (0, 0, 0), -1)
    text(f, "track status in this update", (px + 6, py + 16), WHITE, 0.48)
    for i, tid in enumerate(sorted(status)):
        text(f, f"ID {tid}: {status[tid]}", (px + 6, py + 38 + 20 * i), COLORS.get(tid, GREY), 0.47)
    cap = f"T3b tracking output | source frame {fi} | tracker updates at 4 Hz (frame {key}); boxes held between updates at 24 fps (+{held})"
    text(f, cap, (10, H - 12), WHITE, 0.55)
    return status


def main(out: str = DEFAULT_OUT) -> None:
    root = (Path.cwd() / "outputs").resolve()
    out = str((Path.cwd() / out).resolve())
    if root not in Path(out).parents:                  # output must stay inside ./outputs (no path traversal from the command line)
        raise SystemExit(f"output must be inside {root}")
    by = load()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    try:
        open(out, "xb").close()                         # exclusive create: refuses to overwrite and reserves the path against a concurrent run
    except FileExistsError:
        raise SystemExit(f"refusing to overwrite {out}")
    vw = None
    done = False
    try:
        cap = cv2.VideoCapture(VIDEO)
        cap.set(cv2.CAP_PROP_POS_FRAMES, START)
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, H))
        if not vw.isOpened():
            raise SystemExit(f"cannot open video writer for {out}")
        seen: set[int] = set()
        for fi in range(START, END):
            ok, f = cap.read()
            if not ok:
                raise SystemExit(f"source video ended at frame {fi}")
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
            key = START + ((fi - START) // STEP) * STEP
            st = draw_frame(f, fi, by.get(key, []), seen)
            seen |= {t for t, s in st.items() if not s.startswith("NO ESTIMATE")}
            vw.write(f)
        done = True
    finally:
        if vw is not None:
            vw.release()
        if not done:
            Path(out).unlink(missing_ok=True)           # never leave a partial video that blocks the retry
    print("wrote", out)


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
