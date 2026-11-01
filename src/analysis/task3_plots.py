"""Task 3 figures from saved artifacts only (dev = classical_final_per_image.csv vs the raw S1 reference; eval = final_eval_with_counts.csv)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "outputs/figures"


def main() -> None:
    d = pd.read_csv("experiments/classical_final_per_image.csv")
    ne = d[~d.raw_ref_empty]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist(ne.raw_iou.fillna(0), bins=np.linspace(0, 1, 11), color="#4477aa")
    ax[0].set(title=f"Dev: per-image IoU vs raw S1 reference (n={len(ne)})", xlabel="IoU (empty prediction = 0)", ylabel="images")
    sc = ne.groupby("scene").raw_iou.mean().sort_values()
    ax[1].barh(sc.index, sc.values, color="#cc6677")
    ax[1].set(title="Dev: mean IoU by scene (0 = egg not found)", xlabel="mean IoU")
    ax[2].scatter(ne.raw_fn_px, ne.raw_fp_px, s=12, alpha=.6, color="#117733")
    lim = max(ne.raw_fn_px.max(), ne.raw_fp_px.max())
    ax[2].plot([1, lim], [1, lim], "k--", lw=.8)
    ax[2].set(xscale="log", yscale="log", title="Dev: missed vs extra pixels per image", xlabel="false-negative px", ylabel="false-positive px")
    fig.tight_layout(); fig.savefig(f"{OUT}/task3_dev_iou_scene_errors.png", dpi=130); plt.close(fig)

    s1 = pd.read_csv("experiments/s1_same_protocol_per_image.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, (name, df, col) in zip(ax[:2], (("S1", s1, "n_pred"), ("classical", d, "n_pred"))):
        ct = pd.crosstab(df.expected, df[col]).reindex(index=range(0, 4), columns=range(0, 5), fill_value=0)
        a.imshow(ct.values, cmap="Blues"); a.set_xticks(range(5)); a.set_yticks(range(4))
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                a.text(j, i, int(ct.values[i, j]), ha="center", va="center")
        a.set(title=f"Dev: {name} predicted vs expected count", xlabel="predicted", ylabel="expected (AI-assisted)")
    ids = [f"E{i:02d}" for i in range(14)]
    f1 = []
    for i in ids:
        x = pd.read_csv(f"experiments/classical_{i}_per_image.csv")
        tp, fp, fn = x.rev_inst_tp.sum(), x.rev_inst_fp.sum(), x.rev_inst_fn.sum()
        f1.append(2 * tp / (2 * tp + fp + fn))
    ax[2].bar(ids, f1, color="#999933"); ax[2].set(title="Dev: instance F1 vs R_rev of each candidate config", ylabel="F1")
    ax[2].tick_params(axis="x", rotation=60)
    fig.tight_layout(); fig.savefig(f"{OUT}/task3_dev_counts_and_experiments.png", dpi=130); plt.close(fig)

    e = pd.read_csv("experiments/final_eval_with_counts.csv")
    fig, ax = plt.subplots(figsize=(12, 3.8))
    x = np.arange(len(e)); w = .27
    ax.bar(x - w, e.expected_egg_count, w, label="expected (AI-assisted)", color="#444444")
    ax.bar(x, e.s1_n, w, label="S1", color="#44aa99"); ax.bar(x + w, e.classical_n, w, label="classical", color="#cc6677")
    ax.set_xticks(x); ax.set_xticklabels([i.replace("not_damaged_", "") for i in e.image_id], rotation=60)
    ax.set(title="Evaluation split (22 images): egg count per image", ylabel="eggs"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/task3_eval_counts.png", dpi=130); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
