"""Trajectory figure from the saved tracks.csv files only (no third-party footage, no new experiment)."""
from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluate import GAPS
from .sampling import STEP

OUT = "outputs/figures/video_tracking_trajectories.png"
VARIANTS = (("T0_gated", "T0 gated"), ("T1_gated_plus_two_egg_prior", "T1 gated + two-egg prior"))
PALETTE = ["#0072B2", "#E69F00", "#CC79A7", "#009E73", "#D55E00"]


def load(variant):
    d = defaultdict(list)
    for r in csv.DictReader(open(f"outputs/video/tracking_{variant}/tracks.csv")):
        if r["id"] != "":
            d[int(r["id"])].append((int(r["frame"]), (float(r["x1"]) + float(r["x2"])) / 2))
    return d


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, (var, title) in zip(axes, VARIANTS):
        for (ident, a, b, why) in GAPS:
            ax.axvspan(a, b, color="#999999", alpha=.18)
            ax.text((a + b) / 2, 1860, f"{ident} hidden", ha="center", va="top", fontsize=7)
        ax.axvline(747, color="k", ls="--", lw=1)
        ax.text(752, 560, "hand picks up egg P:\ntrack follows the wrong egg", fontsize=7, va="bottom")
        for tid, pts in sorted(load(var).items()):
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", ms=3, color=PALETTE[(tid - 1) % 5], label=f"track ID {tid}")
            for (f0, x0), (f1, x1) in zip(pts, pts[1:]):   # solid = consecutive samples, dashed = no detection in between (interpolation)
                ax.plot([f0, f1], [x0, x1], "-" if f1 - f0 <= STEP else "--", lw=1, color=PALETTE[(tid - 1) % 5])
        ax.set(title=f"{title}: horizontal egg-box centre per tracked ID (sampled frames, 4 Hz)", ylabel="x centre (px, 1920 wide)", ylim=(0, 1920))
        ax.legend(loc="lower right", fontsize=7, ncol=4)
    axes[1].set_xlabel("frame (24 fps); grey = hand-occlusion gaps of the proxy annotation; dashed = no detection between samples (interpolated line)")
    fig.tight_layout(); fig.savefig(OUT, dpi=130); plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
