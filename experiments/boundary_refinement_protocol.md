# Boundary refinement protocol (written BEFORE any refinement was run; do not edit after results exist)

Status at writing time: no refinement code exists, no refined mask has been generated, no result has been seen.
Scope: development split v2 only (88 images). Evaluation images are rejected by the code and were not inspected.
Labels used (counts, verdicts, scene categories) and the new boundary ratings are AI-assisted and **not verified by a human**.
There are no pixel-level reference masks, so **no IoU against ground truth is reported**.

## Hypothesis
Re-prompting SAM with the bounding box of an existing proposal (`SamPredictor.predict(box=...)`) may give a better egg boundary than the
automatic-generator mask for eggs that are partly occluded, submerged, or have irregular proposal boundaries, without hurting eggs whose
masks are already good.

## Cases (17, all dev; chosen from existing review artifacts; canonical candidate ids from `outputs/review_instances/review_items.csv`)

Targets (10 certain + 1 uncertain). Verdict = effective verdict after second-pass verification.

| review id | candidate id | scene | verdict now | why included |
|---|---|---|---|---|
| D11-c4 | not_damaged_11#4 | vegetation | egg_bad_boundary | leaf occlusion, ragged cutouts |
| D12-c9 | not_damaged_12#9 | vegetation | egg_bad_boundary | leaf occlusion, partial body |
| D13-c2 | not_damaged_13#2 | vegetation | egg | leaf occlusion, ragged edge (verdict `egg`, tests a milder case) |
| D144-c7 | not_damaged_144#7 | marble_counter | egg | overlapping eggs: crescent of the back egg occluded by the front egg |
| D124-c76 | not_damaged_124#76 | steel_pot_water | egg | submerged, stray specks |
| D135-c41 | not_damaged_135#41 | steel_pot_water | egg | submerged, specks at left edge |
| D145-c40 | not_damaged_145#40 | steel_pot_water | egg_bad_boundary | submerged, boxy left edge |
| D42-c12 | not_damaged_42#12 | egg_tray | egg_bad_boundary | partial view cut by a rod |
| D123-c9 | not_damaged_123#9 | white_on_white | egg_bad_boundary | egg behind translucent sheet, upper part only |
| D125-c4 | not_damaged_125#4 | white_on_white | egg_bad_boundary | egg behind translucent sheet, upper part only |
| D117-c23 | not_damaged_117#23 | egg_tray | egg_bad_boundary (**uncertain**) | ambiguous object; reported but **not counted** in any criterion |

Controls (6 correctly selected S1 eggs with verdict `egg`, to check that refinement does no damage; different scenes/difficulties):

| review id | candidate id | scene | why included |
|---|---|---|---|
| D141-c0 | not_damaged_141#0 | plastic_container | large clean egg on a coloured lid |
| D112-c1 | not_damaged_112#1 | white_on_white | low-contrast egg on a white bag |
| D144-c4 | not_damaged_144#4 | marble_counter | front egg overlapping two others |
| D27-c1 | not_damaged_27#1 | ash_rubble | textured, ash-covered egg |
| D134-c3 | not_damaged_134#3 | marble_counter | egg clipped by the image border |
| D129-c1 | not_damaged_129#1 | steel_pot_water | egg floating in water, specular background |

Mask hashes of the original candidates are in `experiments/boundary_refinement_cases.csv` (generated from the same review artifacts).
The case list is closed: no case is added, dropped or swapped after results are seen.

## Method (the only prompts allowed)
- Model: the existing `models/sam_vit_b_01ec64.pth` (sha256 ec2df627...c912), `sam_model_registry["vit_b"]`, `SamPredictor`, GPU if available.
- Prompt: one box = the cached bounding box of the original candidate, xyxy = (x, y, x+w, y+h) from `candidates.jsonl`, clipped to the image.
  No point prompts, no `mask_input`, no negative prompts, no colour or other thresholds.
- `multimask_output=False` (a single mask), mask logits thresholded at SAM's default 0.0.
- Variants, fixed now, each run exactly once:
  - **V1 tight box**: padding 0%.
  - **V2 padded box**: each side expanded by 8% of the box width/height, then clipped.
- Refined masks are stored in `outputs/boundary_refinement/<variant>/masks/` under **new ids** `R1:<image>#<orig_index>` / `R2:<image>#<orig_index>`;
  an original candidate id is never reused for a changed mask; the cache (`outputs/sam_candidates`) and all S1 outputs are read-only (the code refuses to write inside the cache).
- Recorded per refined mask: source image id, original candidate id, original mask sha256, refined mask sha256, prompt box, parameters, SAM predicted IoU,
  runtime, peak GPU memory, code hash, checkpoint hash, torch version.

## Boundary evaluation (qualitative, AI-assisted, blinded as far as practical)
- For each case a panel shows the original and both refined masks as three columns A, B, C in a random order (seed 20261029); the column-to-mask mapping is written to
  `outputs/boundary_refinement/blind_mapping.csv` and is **not opened until all ratings are recorded**. Panels are crops around the egg at larger scale; no text reveals which is which.
- Per column the reviewer records: `rating` 1-3, `covers_intended_egg` (yes/no), `background_leakage` (none/minor/obvious), `occlusion` (plausible/implausible/not_applicable).
  Rubric: **3** boundary follows the visible egg edge closely with no leakage; **2** mostly follows it, minor over/under-coverage; **1** major under/over-coverage, leakage into
  background or other eggs, or a partial/implausible shape.
- After unblinding: for each variant and case, `improved` = refined rating >= original rating + 1, `degraded` = refined rating <= original rating - 1, else `comparable`.
- Descriptive, non-ground-truth numbers only: area ratio refined/original, agreement IoU between original and refined (**agreement, not accuracy**), SAM predicted IoU, runtime, peak GPU.

## Acceptance criteria (per variant; all must hold; evaluated once)
- **A1 targets:** among the 10 certain targets, `improved` in at least 6 and `degraded` in at most 1.
- **A2 controls:** among the 6 controls, `degraded` in 0.
- **A3 leakage:** `obvious` background leakage in at most 1 of the 16 refined masks (10 targets + 6 controls).
- **A4 coverage:** `covers_intended_egg = yes` for at least 90% of those 16 refined masks (at least 15 of 16).
If both variants satisfy all four, V1 (the simpler) is preferred. If neither does, **S1 is unchanged**, the result is documented as negative, and no other prompt,
padding, threshold or point is tried afterwards. D117-c23 is reported only.

## Decision after the experiment
Refinement accepted -> record it as a candidate boundary stage (still not frozen; it would need a separate full dev run). Refinement rejected -> keep S1, proceed with the
manual-selection manifest for flagged dev images and the classical baseline.

## Known limitations declared in advance
Ratings come from the same AI reviewer that made earlier verdicts (blinding is partial: the reviewer can often recognise the original automatic mask style); 16 judged
masks per variant is a small sample; the cases were chosen from known difficult examples and the originals were reviewed earlier by the same reviewer.
