"""Evaluation-split report from the saved run outputs (no method is run here). Dev numbers are read from saved artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.evaluation.metrics import count_metrics

OUT = Path("outputs/eval")


def _mask(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE) > 0


def overlays(ev: pd.DataFrame) -> None:
    (OUT / "overlays").mkdir(exist_ok=True)
    tiles = []
    for r in ev.itertuples():
        img = cv2.imdecode(np.fromfile("data_raw/" + r.path, np.uint8), 1)
        h, w = img.shape[:2]
        row = [img.copy()]
        for name, col in (("S1", (0, 255, 0)), ("classical", (0, 0, 255))):
            m = _mask(OUT / name / "masks" / f"{r.image_id}.png")
            o = img.copy()
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(o, cnts, -1, col, max(2, w // 200))
            row.append(o)
        row = [cv2.resize(x, (360, 270)) for x in row]
        cv2.putText(row[0], r.image_id.replace("not_damaged_", ""), (6, 24), 0, .8, (0, 255, 255), 2)
        cv2.putText(row[1], "S1", (6, 24), 0, .8, (0, 255, 0), 2)
        cv2.putText(row[2], "classical", (6, 24), 0, .8, (0, 0, 255), 2)
        tiles.append(np.hstack(row))
    for n in range(0, len(tiles), 4):
        cv2.imwrite(str(OUT / "overlays" / f"sheet_{n // 4}.jpg"), np.vstack(tiles[n:n + 4]), [1, 88])


def main() -> None:
    res = pd.read_csv("experiments/final_eval_per_image.csv")
    counts = pd.read_csv("data/review/eval_egg_counts.csv")
    d = res.merge(counts[["image_id", "expected_egg_count"]], on="image_id", validate="1:1")
    assert len(d) == 22
    ev = pd.read_csv(OUT / "eval_split.csv").sort_values("image_id")
    overlays(ev)
    e = d.expected_egg_count.to_numpy()
    cm = {"S1": count_metrics(d.s1_n.to_numpy(), e), "classical": count_metrics(d.classical_n.to_numpy(), e)}
    ne = d[~d.agree_ref_empty]
    rec = json.loads((OUT / "RUN_RECORD.json").read_text())
    lines = ["# Final evaluation (22 untouched images, each frozen method run once)", "",
             f"Frozen artifacts: S1_v1 `{rec['s1_artifact_sha256'][:16]}`, classical_v1 `{rec['classical_artifact_sha256'][:16]}`; run `{rec['ran_at']}`; per-image CSV sha `{rec['per_image_csv_sha256'][:16]}`.",
             "Counts: AI-assisted visual counts from the raw images (`data/review/eval_egg_counts.csv`), made after the run and before viewing mask overlays, but the per-image predicted counts had already been printed, so the labeller was NOT blind to them; unverified by a human. There are **no pixel-level labels**; "
             "the only mask reference is S1's own output, so the classical-vs-S1 numbers are agreement with an S1-derived reference, not accuracy, and S1 against itself is not reported.", "",
             "## Egg-count metrics (n = 22; expected total eggs = %d)" % int(e.sum()), "",
             "| method | exact | over | under | mean abs err | max abs err | total predicted |", "|---|---|---|---|---|---|---|"]
    for k, v in cm.items():
        tot = int(d.s1_n.sum() if k == "S1" else d.classical_n.sum())
        lines.append(f"| {k} | {v['exact_rate']:.3f} ({round(v['exact_rate'] * 22)}/22) | {v['over_rate']:.3f} | {v['under_rate']:.3f} | {v['mean_abs_err']:.3f} | {v['max_abs_err']} | {tot} |")
    lines += ["", f"S1 review flag (no candidate passed the gates): {int(d.s1_needs_review.sum())}/22 images ({', '.join(d[d.s1_needs_review].image_id)}); in both the expected count is {', '.join(str(x) for x in d[d.s1_needs_review].expected_egg_count)}.", "",
              "## Classical vs S1 raw masks: agreement only", "",
              f"Images where the S1 mask is non-empty: {len(ne)}/22 (IoU/Dice undefined when the reference is empty; those images are counted in the instance columns).",
              f"- Macro IoU {ne.agree_iou_vs_S1.mean():.3f}, macro Dice {ne.agree_dice_vs_S1.mean():.3f} over {len(ne)} images; classical empty on {int(ne.agree_pred_empty.sum())} of them (IoU 0).",
              f"- Instances at IoU 0.5 vs S1 instances: TP {int(d.inst_tp_vs_S1.sum())}, FP {int(d.inst_fp_vs_S1.sum())}, FN {int(d.inst_fn_vs_S1.sum())} over 22 images.", "",
              "## Runtime", "",
              f"- S1 selector on cached candidates: {rec['s1_selector_total_s']:.1f} s total for 22 images (includes mask writing). Candidate generation (SAM ViT-B, GPU) is not repeated here; the cache run took 1803.6 s for 110 images (16.4 s/image).",
              f"- Classical: median {d.classical_runtime_s.median():.3f} s/image on CPU.", "",
              "## Per image", "", "| image | expected | S1 n | S1 status | classical n | IoU vs S1 | inst TP/FP/FN vs S1 |", "|---|---|---|---|---|---|---|"]
    for r in d.itertuples():
        iou = "n/a" if pd.isna(r.agree_iou_vs_S1) else f"{r.agree_iou_vs_S1:.3f}"
        lines.append(f"| {r.image_id} | {r.expected_egg_count} | {r.s1_n} | {r.s1_status} | {r.classical_n} | {iou} | {r.inst_tp_vs_S1}/{r.inst_fp_vs_S1}/{r.inst_fn_vs_S1} |")
    Path("experiments/final_evaluation_results.md").write_text("\n".join(lines) + "\n")
    d.to_csv("experiments/final_eval_with_counts.csv", index=False)
    print("\n".join(lines[:22]))


if __name__ == "__main__":
    main()
