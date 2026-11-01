# Final evaluation: observed failures (22 evaluation images, viewed after the single run; nothing changed afterwards)

Numbers: `experiments/final_evaluation_results.md` (generated). Overlays: `outputs/eval/overlays/sheet_0.jpg to sheet_5.jpg` (raw | S1 | classical). Counts are AI-assisted and were not labelled blind to the predicted counts (disclosed in the results file).
Both methods are frozen (`configs/frozen/S1_v1.json`, `classical_v1.json`); the evaluation run is recorded once in `outputs/eval/RUN_RECORD.json`.

## Development vs evaluation (reported separately, never pooled)
| | dev (88 images) | eval (22 images) |
|---|---|---|
| S1 exact count | 0.830 (73/88) | 0.864 (19/22) |
| S1 over / under | 0.011 / 0.159 | 0.000 / 0.136 |
| classical exact count | 0.477 (42/88) | 0.545 (12/22) |
| classical over / under | 0.170 / 0.352 | 0.273 / 0.182 |
Dev counts use `data/review/dev_egg_counts.csv`, eval counts `data/review/eval_egg_counts.csv`, both AI-assisted and unverified by a human. The 22-image result is consistent in direction with development (S1 above classical on count agreement); with 22 images the difference between the dev and eval figures of the same method is not interpreted.
No pixel-accuracy ranking is made: there are no human masks, and the only mask reference is S1's own output.

## S1 (3 of 22 counts wrong, all under-counts, no over-counts)
- **Missed with the review flag (10, 122):** a pale egg in green leaves / a glossy egg in white sago: no candidate passed the conservative gates, so the image was flagged `review_no_candidate` rather than given a wrong mask. This resembles the class-B pattern documented on development (SAM proposes the egg but the gates reject it); that was not checked on these two evaluation images.
- **142:** three touching eggs; two were selected, the third (partly behind the other two) was not. An under-count without a flag; the review flag does not catch partial misses.
- **Observation, not a metric:** S1 contours are smooth ellipse-like and in 25, 29, 95 and 164 look slightly larger than the visible egg edge. This is not measured (no ground-truth boundary); the boundary-refinement experiment did not meet its criteria on development.

## Classical (10 of 22 counts wrong: 6 over, 4 under)
- **Empty on white-on-pale objects (126 egg in sago, 31 egg on tile):** whiteness thresholding finds nothing separable.
- **Background regions merged or added (14, 138, 157, 164, 25, 50, 153, 154, 160, 161):** ash, foil, skin, rug, door panels are as bright and unsaturated as the egg, so large background regions pass the area/aspect filters; the egg is often a hole or a neighbour of the region. In 14, 164, 25, 154 and 50 the egg is also present (count right or one extra).
- **Egg plus background merged (10, 107, 122):** a region that contains the egg and sago/leaves; instance mask not usable.
- **Touching eggs merged (142):** the three eggs form one component (watershed split was discarded on development by the pre-registered rule).
- **Partly right (95, 159, 29, 2):** correct egg found but the mask is cut by shading or includes extra area (95 loses the lower shadowed part, 29 splits at a lighting edge).
All of these repeat failure types documented on the development set (`experiments/classical_failure_analysis.md`).

## Limits
- 22 images, 25 eggs; one scene (hand, ash, tile, rug, bowl, sago) often appears once; no confidence intervals are claimed.
- Count labels are not human-verified and were made after the predicted counts were printed.
- S1's selector thresholds are hypotheses chosen on development evidence; the evaluation result does not select or change anything.
