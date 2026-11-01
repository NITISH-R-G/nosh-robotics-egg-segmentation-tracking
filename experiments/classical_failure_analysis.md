# Classical baseline: review of the actual outputs and failure analysis (dev only; no parameter changed after viewing)

Sources: `experiments/classical_results.md` (generated), `experiments/classical_log.csv`, per-image tables `experiments/classical_*_per_image.csv`,
overlays in `outputs/classical/final_overlays/` (each panel: original | R_rev | R_raw = S1 | classical final). Both references are SAM-derived
pseudo-ground-truth; counts and labels are AI-assisted and unverified by a human. Nothing below changes the frozen-nothing status: no configuration is frozen.

## Run verification (executed)
- `python -m src.experiments.classical_experiments`: exit code 0, 279 s; 14 experiments logged (E00-E13), each with 88 dev rows and 88 dev masks; no evaluation id anywhere.
- Independent recomputation of the greedy sequence from the saved per-image files (own implementation of amendment A6) reproduced all 14 decisions and the final configuration `6696e8cab3ba` (E00, E02, E09, E10, E11).
- Independent recomputation of all 15 leave-one-scene-out folds (selection from the 14 training scenes only, then the held-out scene scored): 15 of 15 match `classical_loso.csv`
  (config hash, kept steps, TP/FP/FN, image count). A unit test additionally proves that destroying a held-out scene's results cannot change that fold's selection.
- Raw S1 masks: hashes of all 88 equal the hashes recorded when the reviewed reference was built (unchanged).

## Selected configuration (what the rule produced, not an exception)
Lab whiteness (L minus chroma) foreground, Otsu, 5 px open/close, hole filling, area 1%-60% of the image, aspect ratio <= 2.2, border-touching components >= 5% of the image.
Near-misses decided strictly by the rule: E08 (solidity >= 0.85) had the biggest single F1 jump (0.292 -> 0.417) but lost 0.030 macro IoU (limit 0.010) and was discarded;
E12 (watershed split) gained 0.026 F1 but lost 0.057 exact-count rate (limit 0.030) and was discarded; E04/E06/E07 gains were below 0.02.

## Measured results (S_all, 88 images; pixel numbers are agreement with the named reference, not accuracy)
| | S1 | classical |
|---|---|---|
| count exact / mean abs / max abs | 0.830 / 0.170 / 1 | 0.477 / 0.568 / 2 |
| over / under rate | 0.011 / 0.159 | 0.170 / 0.352 |
| instance P / R / F1 vs R_rev | 1.0 / 0.935 / 0.966 (context only) | 0.442 / 0.393 / 0.416 |
| macro IoU / Dice vs R_rev | 0.924 / 0.925 (context only) | 0.363 / 0.441 |
| macro IoU / Dice vs R_raw | 1.0 / 1.0 (identity) | 0.366 / 0.446 |
| runtime | 16.4 s/image SAM (GPU) + 0.26 s selector | 0.05 s/image median (CPU) |
S1's reference-based numbers are not accuracy: the references are built from S1. S1's independent measure is its verdict-based precision 0.990 / recall 0.876 from the candidate review.
The classical-minus-S1 count-exact difference is negative on S_all, S_no_first_pass and S_clean (-0.352, -0.393, -0.500): the conclusion does not depend on the first-pass-only or unresolved cases.
Classical F1 moves by at most 0.029 across the three subsets.
The right count with wrong instances (a false positive cancelling a miss): 20 images for classical, 0 for S1.

## Leave-one-scene-out
All 15 folds selected the identical configuration, so held-out pooled F1 (0.416 vs R_rev, 0.410 vs R_raw) equals the in-sample F1. This shows the 14-step greedy rule is stable across scenes;
it is not an independent generalisation estimate, because a closed list of discrete choices is rarely changed by dropping one scene. The per-scene failures (F1 vs R_rev) are the generalisation story:
ash_rubble, cloth_white_striped, egg_tray, foil_wrapper, tile_wall and white_on_white all score 0.0; hand_held 0.105; keyboard is the only scene at 1.0 (dark background).

## Overlays inspected (best, worst and category panels; ids in the report)
- **Best (115, 148, 17, 21)**: bright eggs on a dark or mid-tone counter. The mask follows the egg well; it slightly over-extends into pale edge halos and, in 115 and 148, separates touching eggs only by a narrow neck, which the aspect/area filters tolerate because the watershed step is off.
- **White-on-white and pale backgrounds (112, 5, 6, 32, 4, 34, 149)**: empty prediction in 112, 5, 6, 32, 34 and 149; the eggs are not brighter than the cloth, bag, bowl or tile, so no global lightness or whiteness threshold isolates them. In 4 the only prediction is a bright strip of cloth at the bottom-right, so the count is right (1) while the instance is wrong. 34 (egg on a pink bowl) is empty as well, which shows the whiteness map does not rescue coloured backgrounds when the egg is itself grey-white.
- **Backlit egg (24)**: the egg is darker than the wall, so the lightness-based map selects the wall and the finger region instead; this is the failure the unit test documented.
- **Overlapping eggs (144)**: the front-left egg and the right egg are merged with the back egg into one component, covering 3 eggs with 1 mask. (R_raw/S1 also misses the back egg here, a known S1 limit.)
- **Submerged eggs and pots (124, 113)**: in 124 the prediction is a large irregular water/pot region that does not contain the egg as a separate object (the egg blob is bright but merged with bright water); in 113 both eggs are found, but one is merged with a large pale pot region.
- **Foil (118)**: the prediction covers a large part of the foil wrapper and leaves the egg outside the main region, i.e. a bright, reflective, textured region beats the egg.
- **Tile (149, 150, 156, 163)**: empty or wrong; patterned pale tile is as bright and unsaturated as the egg.
- **Steel plate and counter (121)**: the egg on the plate is lost and large plate/counter regions are predicted (the second egg on the counter is segmented well); right count (3) with wrong instances.
- **Earphones, cloth, rubble (16, 27, 41, 52, 62)**: the egg is missed (16, empty) or the prediction is a background region, again with correct counts in several images.

## Why the classical method performs poorly here (evidence, not a new experiment)
1. Brightness and low saturation are properties of the background in most scenes (white bags, tile, cloth, bowls, foil highlights). The log shows no colour space beats E00 by much (E01 -0.002, E03 -0.115, E02 +0.036 F1).
2. Shape filters and a border rule add F1 (0.292 -> 0.416) mainly by removing false components, not by finding more eggs; recall stays below 0.40.
3. The adaptive threshold (E05) collapses count accuracy to 0.0, and the ellipse filter (E13) removes true eggs; neither is a promising direction within this operation set.
4. The failure is structural, not a tuning gap: the search space was closed and is not expanded.

## Limitations and open items
- References are SAM-derived and labels AI-assisted; seven reviewed additions (four first-pass-only) and seven images whose bad-boundary S1 candidate was excluded from the reviewed reference (called 'unresolved' in earlier text; verified in data/review/dev_manual_selection_manifest.csv: 7 exclude rows, 6 confirmed, 1 uncertain); sensitivity subsets show the conclusions do not depend on them.
- Overlay review covered the 6 worst, 4 best and 8 category images (not all 88); the per-image tables hold every image.
- The evaluation split was not used for any decision. It still has no classical or S1 reference outputs from a frozen configuration.
- Not done: freezing of either method, evaluation-split runs, Task 3 plots beyond these tables, video tracking, the PDF report and repository packaging.

## Recommendation
1. **Do not freeze the classical configuration as a competitive method.** Record it as the documented, reproducible baseline (it is the simple configuration selected by the pre-registered rule and it is stable across folds), and report its weakness honestly: F1 0.42, count exact 0.48 vs S1 0.83, 20 right-count-wrong-instance images.
2. For Task 4 (video), the video frames show a title card, a dark brown table and two white eggs with a hand; the classical baseline is plausible there (dark background) but must be checked on development frames of the video first, with detection thresholds set on those frames only.
3. Before any evaluation-split run, decide and freeze S1's selector (still undecided) and the classical configuration (this baseline), then run each once on the 22 evaluation images and report the numbers as measured.
