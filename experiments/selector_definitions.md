# Selector configurations S0-S4 (defined before any run)

All five select from the same cached SAM candidates, using only the same dev images
(`data/split_manifest.csv`, split == dev, 88 images) and the same labels
(`data/review/dev_egg_counts.csv`, AI-assisted image-level counts, not pixel ground truth).
Ranking score is `ellipse_iou * predicted_iou`; duplicates are removed with NMS
(IoU > 0.5 or containment > 0.8). Metrics are identical for every configuration.

| ID | Role | Differences from its parent | Parameters |
|---|---|---|---|
| S0_pilot_filter | simplest baseline | - (the filter used in the 8-image pilot) | area 0.004-0.70, solidity >= 0.90, aspect <= 2.0, ellipse-IoU >= 0.85, pred-IoU >= 0.80, stability >= 0.85 |
| S1_conservative | stricter gates | S0 with tighter thresholds | area 0.01-0.60, solidity >= 0.95, aspect <= 2.0, ellipse-IoU >= 0.93, pred-IoU >= 0.90, stability >= 0.90 |
| S2_conservative_border_rule | handles clipped eggs | S1 + border-touching candidates may have ellipse-IoU >= 0.85 but must cover >= 5% of the image | as S1 + border rule |
| S3_S1_plus_review_flag | review path | S1 + images with no selected candidate are flagged `review_no_candidate` | as S1; **selected masks identical to S1** |
| S4_S1_plus_relaxed_fallback | fallback | S1 + if nothing passes, take the single best candidate under relaxed gates (ellipse-IoU >= 0.80, pred-IoU >= 0.75, stability >= 0.80, solidity >= 0.90), flagged `fallback_top1` | as S1 + fallback |

Known design caveats (stated before results):
- S1 and S3 produce the same selections by construction; S3 only adds a status flag, so its count metrics
  must equal S1's. It is kept because the assignment asks for a review path, not because it is a different selector.
- Thresholds in S1 were chosen after seeing feature values of 8 pilot images (4 of which were later moved
  to dev by the split repair). They are a starting hypothesis, not tuned values; no threshold search has been run.
- S0 on the cache is not byte-identical to the pilot: the pilot generator used predicted-IoU >= 0.80 and
  stability >= 0.85, the cache uses 0.5 / 0.5 (a superset), so candidates that the pilot's generator hid are now visible
  to the selector. The selector gates themselves are unchanged.
- Enforced by tests (`tests/test_selector_experiments.py`): identical images/labels/order for every
  configuration; S1 and S3 select identical candidates; S4 differs from S1 only where S1 selects nothing;
  eval candidate lines are never parsed.
- Logged per run: code hash, config, command, split version + hash, cache `run.json` hash, runtime, status.
- A selector cannot recover an egg for which no suitable candidate was cached (likely for submerged eggs).
- Count metrics can hide cancelling errors (one false positive plus one miss). Instance-level verdicts
  from the review sheets are required before accepting any configuration.
