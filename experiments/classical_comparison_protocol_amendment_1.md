# Amendment 1 to `classical_comparison_protocol.md` (written BEFORE any classical code ran or any classical result was seen)

The original protocol is preserved unchanged (sha256 prefix recorded in `experiments/protocol_amendment_log.csv`). This amendment corrects two errors found by the audit
(`scratch/audit_protocol.py`, run 2026-10-29) and adds the missing definitions. Where the two documents differ, **this amendment governs**.
All labels remain AI-assisted and unverified by a human; both references are pseudo-ground-truth; no pixel ground truth exists.

## A1. Exact images
- Development: the 88 images with `split == dev` in `data/split_manifest.csv` (split version `v2_pilot_exposure_controlled`); their ids are frozen in `experiments/classical_dev_image_ids.txt` with a hash.
- Evaluation: the 22 images with `split == eval`. They are **not used** for any classical design, tuning, threshold, filter or overlay; every classical entry point rejects non-dev ids (tested).

## A2. Reference artifacts (verified on disk) and a correction
- **R_raw** = `outputs/selector_dev/S1_conservative/masks/<image_id>.png` (88 files, {0,255}, union of the S1-selected candidates; empty in 13 images). The name `outputs/reference_raw_dev/` does not exist.
- **R_rev** = `outputs/reference_reviewed_dev/masks/<image_id>.png` (88 files; empty in 7 images). Separate artifact; R_raw is never modified.
- **Correction:** the original text says R_raw stores "one union mask plus instance labels". It does not. **Instance references are reconstructed**:
  R_raw instances = the individual cached candidate masks listed in `selected_idx` of `experiments/selector_S1_conservative_20261029T132403_per_image.csv`;
  R_rev instances = those plus the candidates added by `data/review/dev_manual_selection_manifest.csv`. A test asserts that the union of the reconstructed instances equals the stored union mask bit for bit.
- Seven manual additions keep their status visible: confirmed (124, 135, 144) and first-pass-only (13, 20, 52, 116). Seven images remain empty in R_rev because no clean candidate exists
  (11, 12, 42, 117, 123, 125, 145; 117 is uncertain).

## A3. Sensitivity subsets (every table is reported for all three)
- **S_all**: all 88 dev images.
- **S_no_first_pass**: S_all minus the 4 images whose reviewed addition is first-pass-only (13, 20, 52, 116) -> 84 images.
- **S_clean**: S_all minus the 7 images with a manual addition and the 7 images left unresolved (11, 12, 42, 117, 123, 125, 145) -> 74 images where R_raw and R_rev are identical and no known egg is missing from the references.
A conclusion is called robust only if the ranking of configurations and the sign of the classical-vs-S1 differences are the same on all three subsets.

## A4. Metrics, matching and error definitions (identical for the classical method and for S1)
- Pixel level (against R_raw and R_rev, separately): IoU, Dice, precision, recall, FP/FN pixels; macro average over images with a non-empty reference, and pooled over pixels. Images with an empty reference are reported separately
  (predicted foreground there is a false positive area); an empty prediction with a non-empty reference has IoU 0. These are **agreement with the named reference**, never accuracy.
- Instances: predicted instances = connected components of the predicted mask after the pipeline's own splitting step (classical) or the selected candidates (S1). Reference instances as in A2.
  Matching: Hungarian assignment on mask IoU, a pair counts only if IoU >= 0.5; unmatched predictions are false positives, unmatched reference instances are misses. A merged component of touching eggs can match at most one reference egg.
  Instance precision/recall/F1 are computed against R_rev instances and against R_raw instances separately. Counts of FP and misses are reported per image so a false positive can never hide a miss.
- Counts: predicted instance count vs `data/review/dev_egg_counts.csv` (AI-assisted, uncertain): exact accuracy, mean/median/max absolute error, over/under rates. For S1 the same functions are applied to its selected candidates.
  S1's instance scores against R_rev/R_raw are context only (the references are built from S1); S1's verdict-based precision/recall from the review remain the independent-label measure for S1 and are quoted as such.
- Runtime: classical = `time.perf_counter` around the pipeline function (decoded image in memory, no disk I/O), per image, with median, mean and total over the 88 images. S1 = SAM candidate generation per image from `outputs/sam_candidates/run.json`
  (total 1803.6 s for 110 images on the RTX 3050) plus the selector time from `experiments/selector_log.csv`; both are reported, not combined into one number.

## A5. Classical method and the complete, closed search space
Operations (each justified in the code docstring): foreground map F (higher = more egg-like) -> threshold -> morphology -> connected components -> component filters -> optional touching-egg split. Eggs in this data are white or cream and mostly brighter and
less saturated than their surroundings, which motivates F; that assumption is expected to fail on white-on-white, foil, pale tile and cloth, and the experiments are meant to measure by how much.
Baseline **E00**: F = Lab lightness L, global Otsu threshold, opening then closing with a 5 px elliptical kernel, fill holes, keep components with area in [0.5%, 60%] of the image, no other filter, no split.
Experiments run **in this order, each a single change from the current best configuration (greedy)**; this list is closed and no experiment may be added afterwards:

| id | single change | parameter values |
|---|---|---|
| E00 | baseline | as above |
| E01 | F = HSV value V | - |
| E02 | F = whiteness = L - chroma(a,b) (Lab) | weight 1 |
| E03 | F = inverted HSV saturation | - |
| E04 | Gaussian blur before Otsu | sigma 2 px |
| E05 | adaptive Gaussian threshold instead of Otsu | block 151, C = -5 |
| E06 | larger morphology kernels | open 9, close 9 |
| E07 | smaller morphology kernels | open 3, close 3 |
| E08 | solidity filter | >= 0.85 |
| E09 | aspect-ratio filter (min-area rectangle) | <= 2.2 |
| E10 | larger minimum area | >= 1% of the image |
| E11 | border rule | components touching the border need area >= 5% of the image |
| E12 | distance-transform watershed split | peak height 0.4 x max distance |
| E13 | ellipse-fit filter | component IoU with its fitted ellipse >= 0.85 |
Colour spaces E01-E03 are judged against E00 and the best one is carried forward; E04-E13 are judged against the current best.

## A6. Decision rule and tuning procedure (development images only)
- Selection score = **instance F1 against R_rev** (S_all). A change is **kept** only if score improves by >= 0.02 **and** macro IoU against R_rev does not fall by more than 0.01 **and** the exact-count rate does not fall by more than 0.03.
  Otherwise it is discarded (ties go to the simpler configuration). Every trial, kept or discarded, is logged.
- Leave-one-scene-out: the whole greedy sequence is re-run 15 times, each time on the 14 other scenes (the 15 scenes of `data/review/dev_scene_categories.csv`), and the resulting configuration is applied to the held-out scene.
  The pooled held-out instance F1 is reported next to the in-sample F1 as the generalisation estimate across backgrounds. No scene-specific parameters exist.
- Per-image predictions, masks, configuration and metrics of every experiment are saved under `outputs/classical/<experiment_id>/` and `experiments/classical_*`; the final configuration is hashed.

## A7. Failure analysis
Scenes map to the failure modes named in the task: white_on_white (white-on-white), foil_wrapper (reflective foil), tile_wall (patterned tile), cloth_white_striped and patterned_fabric (cloth), steel_pot_water (steel pots; the `submerged` tag marks submerged eggs),
and the tags leaf_occluded, touching_or_adjacent, border_clipped (occlusion/touching/clipping). Metrics are broken down by scene and by these tags; the worst dev images of the final configuration are inspected as overlays.

## A8. What is not decided here
No threshold of "good enough" is set for the classical method: it is reported as measured. If it performs poorly the reasons are documented and the search space is **not** expanded. Nothing about S1 is frozen.


## Amendment 1b (clarification of E12, written before any classical experiment ran)
E12's "distance-transform watershed split, peak height 0.4 x max distance" is implemented as: seeds = the parts of each component whose distance
transform is >= (1 - 0.4) x the component's maximum distance (0.6 x max); flooding on -distance within the mask (skimage `watershed`).
Two touching eggs are separated only if the neck between them is lower than 0.6 x the peak. This is the level-set reading of "peak height"; skimage's `h_maxima`
was tested and rejected because it keeps both of two equal peaks regardless of h, and OpenCV's `cv2.watershed` was rejected because it floods on colour gradients.
No parameter value changed (0.4). Discovered while writing the unit tests; no experiment had been run.
