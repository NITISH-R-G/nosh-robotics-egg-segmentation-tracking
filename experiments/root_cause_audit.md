# Root-cause audit: apparent false positives on the round-2 review sheets

Scope: development split v2 only (88 images). No evaluation image or evaluation candidate was opened, parsed or rendered; every
diagnostic entry point rejects non-dev ids (tests below). Labels used (counts, candidate verdicts) are AI-assisted and unverified by a
human; they are not pixel ground truth and no IoU is reported.

## Verdict

1. **The contours in the screenshots were not S1 selections.** They were candidates that only the relaxed experiments (E1/E2) admitted.
   S1's own selections (the eggs) were not drawn, because the round-2 sheets were generated with `--only-unreviewed`, which removed every
   already-reviewed candidate from the picture and gave no indication of that. This is a **display defect (class C2)**, not a wrong mask.
2. **No rendering, indexing, coordinate or serialization bug was found (class C1: none).** Mask bytes, ids, hashes, selections and coordinates
   were verified independently for all 88 dev images (table below).
3. **The genuine selector problem is different and smaller:** 14 of 88 images have a missed egg. For all 14 a reviewed egg candidate exists in the
   cache, so these are **selection failures (class B)**; none is a proposal failure (class A) among the reviewed candidates.
4. The freeze I had written just before your message is withdrawn (see Process note).

## What the screenshots displayed (executed: `scratch/what_was_shown.py`, outputs in `outputs/diagnostics/`)

| image | S1 selected (verdict) | contours on the round-2 sheet (config, verdict) | overlap with S1 |
|---|---|---|---|
| 141 | D141-c0 (egg) | D141-c11 (E1, not_egg: blue bowl) | none |
| 144 | D144-c0, c4 (egg, egg) | D144-c7 (E1, egg: back egg, crescent) | none |
| 149 | D149-c0 (egg) | D149-c2 (E1, not_egg: tile/plate wedge) | none |
| 163 | D163-c0 (egg) | D163-c3 (E1, not_egg: wall and bowl) | none |
| 112 | D112-c1 (egg) | D112-c53 (E2, not_egg: bag region) | none |
| 118 | D118-c0 (egg) | D118-c1, c6 (E1, not_egg: dark table, foil label) | none |
| 131 | D131-c2 (egg) | D131-c4, c87 (E1/E2, not_egg: foil regions) | none |
| 139 | D139-c0 (egg) | D139-c5 (E1, not_egg: container body) | none |
| 145 | none (flagged) | D145-c40 (E2+E5, egg_bad_boundary) | none |
| 135 | none (flagged) | D135-c41 (E2+E5, egg) | none |

All ten are dev images. For 141, 149, 163, 112, 118, 131, 139 and both eggs in 144, S1's selected mask is the egg and was simply not drawn.
145 and 135 are real S1 misses whose good proposals (c40, c41) were admitted only by the relaxed gates.

## Mapping and coordinate verification (executed: `python -m src.experiments.diagnose`, all 88 dev images, `outputs/diagnostics_all_dev/`)

| check | what it proves | result |
|---|---|---|
| I1 cached mask vs manifest | area, bbox (SAM-computed at generation time) and centroid agree with the unpacked mask at that index | 272/272 |
| I2 review item vs cache | review-item mask hash equals the hash of the cached mask; review id = D&lt;img&gt;-c&lt;index&gt; maps to the same candidate | 272/272 |
| I3 live re-selection | re-running S1 on the cached features reproduces the stored selected ids | 88/88 |
| I4 saved union | `outputs/selector_dev/S1_conservative/masks/<id>.png` equals the union of the cached masks of the selected ids, bit for bit | 88/88 |
| I5 contour frame | contours extracted for display lie on the mask boundary in the same (x, y) frame | 272/272 |
| I5 prompt point | SAM's prompt point is within 5 px of its mask | 272/272 |

Coordinate system, tested against alternatives (executed: `scratch/point_rate.py`, 11,306 dev candidates): the prompt point lies inside its mask
for 90.1% of candidates, versus 7.3% if masks were transposed and 3.7% if y-flipped. The 9.9% that fall outside are a median of 1 px away (max 5 px),
which is SAM boundary behaviour. SAM's bbox (computed from the mask when it was generated) matches the bbox recomputed from the stored bits
within 1 px for every candidate, so storage did not change the coordinate frame either. Candidate index 0..N-1 in the NPZ is the same index in
`candidates.jsonl`, in the per-image selection CSVs, in `review_items.csv` and in the sheets (I1, I2, I3, tested with swapped masks).

Two of my own audit checks were wrong and were fixed (no pipeline impact): the boundary check false-alarmed on border-touching masks (OpenCV
replicates border pixels in the gradient), and "prompt point strictly inside" is stricter than SAM guarantees.

## Failure classes for the ten requested images

| image | class | evidence |
|---|---|---|
| 141, 149, 163, 112, 118, 131, 139 | **C2 display only; S1 correct** | S1 mask is a reviewed egg; round-2 sheet omitted it (overlap empty) |
| 144 | **B** for the 3rd egg; S1 correct for the other two | D144-c7 reviewed `egg` (visible crescent of the back egg), rejected by S1: solidity 0.83 < 0.95, ellipse-IoU 0.78 < 0.93 |
| 145 | **B** | D145-c40 exists (egg_bad_boundary, boxy left edge); rejected: pred-IoU 0.80 < 0.90, stability 0.79 < 0.90, ellipse-IoU 0.90 < 0.93 |
| 135 | **B** | D135-c41 exists (egg); rejected: solidity 0.89 < 0.95, ellipse-IoU 0.91 < 0.93, pred-IoU 0.83 < 0.90 |

Class A (no suitable proposal) was looked for and not found: classification used reviewed masks, not bounding-box overlap alone. Across all
88 dev images: 74 covered, 14 class B (11, 12, 13, 20, 42, 52, 116, 117, 123, 124, 125, 135, 144, 145), 0 class A. Several of the 14 only have
approximate proposals (117 ambiguous, 42/123/125/145 bad boundary).

## Code paths responsible

| defect | where | status |
|---|---|---|
| partial sheets hid reviewed (incl. S1-selected) masks, no config tag, no banner | `src/experiments/review_pool.py`: `--only-unreviewed` branch in `main()` and `render_panel()` (before this audit) | fixed: reviewed items drawn gray with verdict, configs shown per contour, "PARTIAL" banner |
| candidate id to mask identity was never verified; wrong index could be relabelled silently | `build_items()` in the same file | fixed: `verify_identity()` raises on area/bbox/centroid mismatch, bad index, missing manifest row |
| `build_items` parsed every line of `candidates.jsonl`, including evaluation rows | same file | fixed: `read_candidates()` in `src/segmentation/cache_candidates.py` skips non-requested lines before JSON parsing; `load_dev` uses it too |
| `fillna("")` on an all-empty tag column | `selector_experiments.run_config` | fixed (warning removed) |

## Selection logic audit (executed: `scratch/logic_audit.py`)

- S1 = hard gate on area, aspect, solidity, ellipse-fit, SAM predicted-IoU and stability, then NMS, and **every** survivor is kept. It does not
  assume the top-ranked candidate is an egg. 143 dev candidates pass the gate; 100 are selected; all 43 others overlap a selected mask under the NMS rule.
- **Geometry does all the discriminating; SAM confidence does not.** Of 159 reviewed non-eggs only 1 passes the shape+area gate (the earphone bud),
  but 117 (74%) pass the confidence gate. Non-egg ellipse-IoU: median 0.866, 90th percentile 0.896, max 0.968; egg: median 0.977, min 0.740. Solidity
  overlaps even more (non-egg median 0.964).
- 13 of 113 reviewed egg candidates fail S1's shape gate (occlusion, partial views, motion blur). Relaxing shape admits background that is also
  smooth and convex: E1 added 32 non-egg regions (7 keyboard keys, cloth, tile and tray cavities, foil, steel plate, wall).
- **The "most ellipse-like / top-ranked candidate is an egg" assumption is false.** In the 11 S1-flagged images that have a relaxed candidate, the
  top-ranked one is `not_egg` in 3 (11, 12, 135), `egg_bad_boundary` in 4 and a clean `egg` in 4. This is exactly what the S4 top-1 fallback relied on.
- No colour threshold was added or tried.

## Regression tests (executed: `python -m pytest -q -p no:cacheprovider -W default` -> **116 passed, 0 failed, 0 skipped, 0 warnings**)

`tests/test_audit_regressions.py` (11 tests, synthetic non-square fixtures so transposition or index mix-ups change the outcome):
selected ids map to the exact cached masks and saved union; overlay contour sits in the image (x, y) frame without resizing; the contour check fails
for transposed, shifted and flipped overlays but passes border-touching masks; the prompt-point distance separates the right mask from another egg's;
ids stay consistent through selection, per-image CSV, review items, provenance and rendering; swapped cached masks raise instead of being relabelled;
out-of-range and unknown candidates raise; context rendering keeps reviewed items visible; `diagnose.assert_dev` and `diagnose.main` reject evaluation
images before touching the cache; `read_candidates` and `load_dev` never parse evaluation lines; `build_items` rejects evaluation runs.

## Is SAM + selector salvageable?

- **Proposal generation: yes.** Every egg that S1 misses has a reviewed proposal in the cache (class B only).
- **Conservative selection: yes as a high-precision stage** (S1: precision 0.990, recall 0.876 on AI-assisted verdicts; 99 of 100 selections are
  reviewed eggs, none has a bad-boundary verdict). The integrity audit found nothing that undermines those numbers.
- **Geometric selection alone cannot recover the remaining 14.** Shape relaxation admits background (E1) and confidence carries no signal (74% of
  non-eggs are confident). The selection method must change for these cases; more thresholds are not justified.

## Recommended smallest next step (dev-only, not yet run)

1. **Measure before choosing:** compute simple egg-appearance descriptors for the 272 reviewed dev candidates (inside versus a surrounding ring: brightness,
   saturation, local contrast, interior texture) and report class separation **per background and lighting category** (white bag, tile, steel pot, cloth, hand,
   backlit). Only a descriptor that separates eggs from non-eggs on the hard backgrounds is eligible; colour thresholds are not set by hand.
2. **If no simple descriptor works across backgrounds,** run a separate, controlled dev-only comparison of an egg-specific scorer (for example CLIP
   zero-shot on candidate crops) against S1 on the same 88 images, after verifying its API, dependency and licence. Apply it only to candidates passing a
   relaxed geometry gate in the 13 images where S1 selects nothing, so S1's precision is preserved.
3. **Parallel, evaluation-safe path:** a reviewer chooses among cached candidates for the flagged dev images (manual selection manifest with image id,
   candidate id, mask hash, reviewer status, reason). Raw automatic masks stay separate. Evaluation images cannot be reviewed this way, so any evaluation
   image where S1 selects nothing stays unsegmented in the automatic reference and must be reported as such.

Do not freeze a selector until step 1 or 2 shows a measured improvement on reviewed dev candidates without degrading precision or boundary quality.

## Process note (disclosure)

Immediately before your message I wrote `configs/selector_frozen.json` and a decision document, based on the round-2 results. That freeze was premature and
is withdrawn: the file now lives at `configs/withdrawn/selector_frozen_WITHDRAWN_20261029.json`, `experiments/selector_decision.md` carries a banner,
`experiments/selector_log.csv` has an appended `FREEZE WITHDRAWN` row, and no reference masks were generated from it. All earlier experiment results and
decision rows are preserved but are not accepted. The round-2 verdicts themselves were assigned per candidate id from the displayed masks and are consistent
with the diagnostic overlays I viewed in this audit (141, 144, 149, 135). I did not re-inspect the other 29 round-2 verdicts against
full-context overlays, so they stand only pending your review (the per-image panels for all 88 dev images are in `outputs/diagnostics_all_dev/`).
