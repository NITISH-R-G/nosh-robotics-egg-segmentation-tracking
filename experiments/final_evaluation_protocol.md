# Final evaluation protocol (written BEFORE freezing and before any evaluation-split run)

## What is evaluated
Two separately frozen methods (`configs/frozen/S1_v1.json`, `configs/frozen/classical_v1.json`), each run **once** on the 22 evaluation images of `data/split_manifest.csv` (v2, sha prefix d971faec).
No threshold, filter or selector rule may be changed after freezing; a change = new version file + logged reason, and a second evaluation run is refused by `src/experiments/final_eval.py` (`outputs/eval/RUN_RECORD.json`).
The evaluation split has not been used for any decision so far: no evaluation image was inspected and no method was run on it (the cached SAM candidates for all 110 images exist from the Option-1 cache run, which was parameter-free).

## What is NOT claimed
- The methods are **not ranked by pixel accuracy**: no human-made masks exist; the only reference for evaluation images is S1's own raw masks, so "S1 vs S1" is trivially perfect and is not a result.
- Development numbers (88 images) and evaluation numbers (22 images) are reported separately and never pooled; no model or threshold is selected from evaluation results.

## What is reported on the evaluation split
1. Per-image detected egg count for both methods; S1 review-flag rate (images where nothing passed the gates).
2. Agreement of the classical masks with S1's raw masks (pixel IoU/Dice, instance match at IoU 0.5), labelled "agreement with an S1-derived reference". Empty-mask handling as in `src/evaluation/metrics.py` and exact denominators.
3. Runtime: S1 selector time (SAM candidate generation time from the cache run is quoted separately, GPU), classical per-image CPU time.
4. Count metrics (exact / mean abs / over / under) ONLY IF eval count labels exist. They will be created **after** the runs from the raw images only, as AI-assisted visual counts (same procedure and caveats as `data/review/dev_egg_counts.csv`: unverified by a human, two passes on ambiguous images), stored in `data/review/eval_egg_counts.csv` and flagged post-freeze. They must not influence any method.
5. Failure cases: all 22 overlays are viewed after the runs; failures are listed and categorised, not fixed.

## Comparison to development
Development metrics come from existing saved artifacts (`experiments/classical_results.md`, `experiments/selector_results.md`, `experiments/fallback_evaluation.md`) and are copied unchanged next to the evaluation table. Any difference between dev and eval is described, not explained away; with 22 images no significance claim is made.
