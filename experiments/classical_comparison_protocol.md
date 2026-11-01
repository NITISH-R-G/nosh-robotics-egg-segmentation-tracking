# Classical baseline and pretrained-vs-classical comparison protocol (written BEFORE any classical experiment ran)

Status at writing time: no classical segmentation code exists and no classical result has been seen. Development split v2 only (88 dev / 22 eval).
The evaluation split is not used for any design, tuning or threshold choice, and its images are not inspected.
All labels (counts, candidate verdicts, scene categories, manual selections) are AI-assisted and not verified by a human. There are no pixel-level reference masks.

## The comparison problem, stated honestly
The assignment treats the Task 1 (SAM) masks as ground truth for the classical method. Consequences that this protocol makes explicit:
1. Pixel agreement of the classical masks with a SAM-derived reference measures **agreement with a pretrained model**, not biological accuracy.
2. The pretrained pipeline cannot be scored against itself with the same pixel metric: its raw masks are the reference. The reviewed reference
   (raw S1 plus manual additions) differs from the raw one in only 7 of 88 dev images, so S1's agreement with it is inflated by construction.
3. Therefore pretrained and classical are compared on measures that do not use SAM masks as truth (image-level egg counts and instance detection
   against the AI-assisted count labels), and the SAM-referenced pixel metrics are reported for the classical method alone, with S1-vs-reviewed shown only as context.

## References (dev only; both pseudo-ground-truth)
- **R_raw**: raw S1 masks `outputs/selector_dev/S1_conservative/masks` (the required Task 1 comparison; 13 images are empty because S1 selected nothing, and the third egg of image 144 is missing).
- **R_rev**: `outputs/reference_reviewed_dev` = R_raw plus the 7 clean-boundary additions of `data/review/dev_manual_selection_manifest.csv`
  (7 images with a known egg but no clean candidate remain empty). Reported separately from R_raw, never merged into it.
- The selector is **not frozen**; R_raw can change if the selector decision changes, so every result records the hash of the reference directory it used.

## Metrics (defined once, used for every experiment; implemented and unit-tested before use)
- Pixel level against a reference: IoU, Dice, precision, recall, FP/FN pixel counts. Reported as **macro average over images** and as **pooled over pixels**.
  Images whose reference is empty are excluded from the macro IoU/Dice/recall and reported separately (classical foreground pixels there are false positives by definition);
  images whose prediction is empty with a non-empty reference have IoU 0 (not dropped). No metric divides by zero silently.
- Instance level: connected components of each mask, matched to reference components with the Hungarian algorithm on mask IoU, threshold 0.5; precision, recall, F1.
  Touching eggs merged into one component are counted as one instance and reported as a known limitation (this is the reference's own convention too: R_raw stores one union mask plus instance labels).
- Count level against `data/review/dev_egg_counts.csv`: exact accuracy, mean/median/max absolute error, over/under rates. Count accuracy is never the only criterion.
- Breakdown by the 15 scene categories (`data/review/dev_scene_categories.csv`) and by lighting tercile; group-aware: classical parameters are chosen on all dev images but their stability is
  reported **leave-one-scene-out** (parameters re-chosen without the held-out scene); no scene-specific parameters.
- Runtime per image and total.

## Classical experiments (dev only)
- Start from the simplest interpretable baseline (one colour space, one global threshold, morphology, largest/size-filtered components), then change one factor at a time:
  colour space (RGB / HSV / Lab), global vs adaptive threshold, morphology, component filters (area, aspect, solidity, circularity, border contact), watershed for touching eggs only if touching failures are measured.
- Append-only log `experiments/classical_log.csv` with the contract's fields (experiment id, timestamp, code version, dataset/split version, seed, preprocessing, colour space, threshold method,
  parameters, morphology, filters, motivation, dev metrics against **both** references, runtime, observed failures, decision, next change). Failed trials are kept.
- Selection of the final configuration uses dev metrics and qualitative inspection of dev overlays only, then the configuration is frozen (hash) before the evaluation split is ever touched.
- Acceptance is by macro IoU **and** instance F1 **and** count error, never one of them alone; overlays of the worst dev images are inspected before any conclusion.

## What the final comparison table will contain
| row | counts vs labels | instance detection vs labels | pixel agreement with R_raw / R_rev |
|---|---|---|---|
| S1 (pretrained + conservative selector) | yes | yes (verdict-based, from the review) | context only (R_raw is S1; vs R_rev shows how much manual review changed it) |
| classical (frozen) | yes | yes (matched to R_rev instances and to the labels) | yes, both references, reported separately |
No statement of "better" is made from SAM-referenced pixel metrics alone.

## Declared limitations
SAM-derived pseudo-ground-truth; AI-assisted labels by one reviewer; 88 development images from 15 scenes with several images sharing backgrounds; evaluation references can only be raw automatic masks
(manual review of evaluation images is not allowed), so evaluation-split results will not be comparable to the reviewed-reference dev results.
