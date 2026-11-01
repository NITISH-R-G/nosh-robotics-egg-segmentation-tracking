# Final evaluation (22 untouched images, each frozen method run once)

Frozen artifacts: S1_v1 `a6e46871795be95e`, classical_v1 `d8c827fe7d9d12ff`; run `20261101-130625`; per-image CSV sha `88c11c96393b3b0a`.
Counts: AI-assisted visual counts from the raw images (`data/review/eval_egg_counts.csv`), made after the run and before viewing mask overlays, but the per-image predicted counts had already been printed, so the labeller was NOT blind to them; unverified by a human. There are **no pixel-level labels**; the only mask reference is S1's own output, so the classical-vs-S1 numbers are agreement with an S1-derived reference, not accuracy, and S1 against itself is not reported.

## Egg-count metrics (n = 22; expected total eggs = 25)

| method | exact | over | under | mean abs err | max abs err | total predicted |
|---|---|---|---|---|---|---|
| S1 | 0.864 (19/22) | 0.000 | 0.136 | 0.136 | 1 | 22 |
| classical | 0.545 (12/22) | 0.273 | 0.182 | 0.500 | 2 | 26 |

S1 review flag (no candidate passed the gates): 2/22 images (not_damaged_10, not_damaged_122); in both the expected count is 1, 1.

## Classical vs S1 raw masks: agreement only

Images where the S1 mask is non-empty: 20/22 (IoU/Dice undefined when the reference is empty; those images are counted in the instance columns).
- Macro IoU 0.315, macro Dice 0.439 over 20 images; classical empty on 2 of them (IoU 0).
- Instances at IoU 0.5 vs S1 instances: TP 6, FP 20, FN 16 over 22 images.

## Runtime

- S1 selector on cached candidates: 4.8 s total for 22 images (includes mask writing). Candidate generation (SAM ViT-B, GPU) is not repeated here; the cache run took 1803.6 s for 110 images (16.4 s/image).
- Classical: median 0.056 s/image on CPU.

## Per image

| image | expected | S1 n | S1 status | classical n | IoU vs S1 | inst TP/FP/FN vs S1 |
|---|---|---|---|---|---|---|
| not_damaged_10 | 1 | 0 | review_no_candidate | 1 | n/a | 0/1/0 |
| not_damaged_107 | 1 | 1 | ok | 1 | 0.292 | 0/1/1 |
| not_damaged_122 | 1 | 0 | review_no_candidate | 1 | n/a | 0/1/0 |
| not_damaged_126 | 1 | 1 | ok | 0 | 0.000 | 0/0/1 |
| not_damaged_128 | 2 | 2 | ok | 1 | 0.407 | 1/0/1 |
| not_damaged_138 | 1 | 1 | ok | 1 | 0.220 | 0/1/1 |
| not_damaged_14 | 1 | 1 | ok | 2 | 0.133 | 0/2/1 |
| not_damaged_142 | 3 | 2 | ok | 1 | 0.298 | 0/1/2 |
| not_damaged_153 | 1 | 1 | ok | 1 | 0.256 | 0/1/1 |
| not_damaged_154 | 1 | 1 | ok | 1 | 0.342 | 0/1/1 |
| not_damaged_157 | 1 | 1 | ok | 2 | 0.282 | 0/2/1 |
| not_damaged_159 | 1 | 1 | ok | 1 | 0.711 | 1/0/0 |
| not_damaged_160 | 1 | 1 | ok | 1 | 0.116 | 0/1/1 |
| not_damaged_161 | 1 | 1 | ok | 1 | 0.235 | 0/1/1 |
| not_damaged_164 | 1 | 1 | ok | 2 | 0.418 | 1/1/0 |
| not_damaged_2 | 1 | 1 | ok | 2 | 0.575 | 1/1/0 |
| not_damaged_25 | 1 | 1 | ok | 2 | 0.278 | 0/2/1 |
| not_damaged_29 | 1 | 1 | ok | 1 | 0.591 | 1/0/0 |
| not_damaged_30 | 1 | 1 | ok | 1 | 0.151 | 0/1/1 |
| not_damaged_31 | 1 | 1 | ok | 0 | 0.000 | 0/0/1 |
| not_damaged_50 | 1 | 1 | ok | 2 | 0.133 | 0/2/1 |
| not_damaged_95 | 1 | 1 | ok | 1 | 0.855 | 1/0/0 |
