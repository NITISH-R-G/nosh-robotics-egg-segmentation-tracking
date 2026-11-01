"""Comparison figure T1 vs T3 from the saved tracks.csv files only (no footage). Observed complete = filled dots, observed partial = triangles,
predicted = hollow dashed, uncertain association = red diamond, unassigned fragments = grey crosses."""
from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluate import GAPS
from .sampling import STEP

OUT = "outputs/figures/video_tracking_T1_vs_T3b.png"
PAL = ["#0072B2", "#E69F00", "#CC79A7", "#009E73", "#D55E00"]


def load(variant):
    rows = list(csv.DictReader(open(f"outputs/video/tracking_{variant}/tracks.csv")))
    for r in rows:
        r["cx"] = (float(r["x1"]) + float(r["x2"])) / 2 if r["x1"] != "" else None
        r["frame"] = int(r["frame"])
    return rows


def panel(ax, rows, title, new_format):
    for (ident, a, b, why) in GAPS:
        ax.axvspan(a, b, color="#999999", alpha=.16)
        ax.text((a + b) / 2, 1880, f"{ident} hidden (annotation)", ha="center", va="top", fontsize=6.5)
    by = defaultdict(list)
    for r in rows:
        if r["id"] != "" and r["cx"] is not None and (not new_format or r["state"] == "observed"):
            by[int(r["id"])].append((r["frame"], r["cx"], r.get("kind", "complete")))
    for tid, pts in sorted(by.items()):
        c = PAL[(tid - 1) % 5]
        for (f0, x0, _), (f1, x1, _) in zip(pts, pts[1:]):
            ax.plot([f0, f1], [x0, x1], "-" if f1 - f0 <= STEP else "--", lw=1, color=c)
        ax.scatter([p[0] for p in pts if p[2] == "complete"], [p[1] for p in pts if p[2] == "complete"], s=9, color=c, label=f"ID {tid} observed")
        part = [p for p in pts if p[2] == "partial"]
        if part:
            ax.scatter([p[0] for p in part], [p[1] for p in part], s=22, marker="^", color=c, label=f"ID {tid} partial fragment")
    if new_format:
        pr = [r for r in rows if r["state"] == "predicted"]
        for r in pr:
            ax.scatter([r["frame"]], [r["cx"]], s=16, facecolors="none", edgecolors=PAL[(int(r["id"]) - 1) % 5], linewidths=.8)
        if pr:
            ax.scatter([], [], s=16, facecolors="none", edgecolors="k", label="predicted (not observed)")
        un = [r for r in rows if r["state"] == "observed" and r["uncertain"] == "1"]
        ax.scatter([r["frame"] for r in un], [r["cx"] for r in un], s=34, marker="D", facecolors="none", edgecolors="red", linewidths=1.2, label="uncertain association")
        ua = [r for r in rows if r["state"] == "unassigned"]
        ax.scatter([r["frame"] for r in ua], [r["cx"] for r in ua], s=14, marker="x", color="#777777", label="unassigned fragment (false: caption arrow)")
    ax.axvline(747, color="k", ls=":", lw=1)
    ax.set(title=title, ylabel="x centre (px of 1920)", ylim=(0, 1920))
    ax.legend(loc="upper right", fontsize=6, ncol=2, framealpha=.9)


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True)
    panel(axes[0], load("T1_gated_plus_two_egg_prior"), "T1 (previous): complete detections only; ID 2 jumps to the wrong egg at the pick-up (dotted line ~ frame 747)", False)
    panel(axes[1], load("T3b_kalman_partial_bounded"), "T3b (Kalman + partial fragments, bounded prediction): fragments carry both identities through the hand", True)
    axes[1].set_xlabel("source frame (24 fps, sampled every 6th); grey = hand-occlusion gaps of the proxy annotation")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
