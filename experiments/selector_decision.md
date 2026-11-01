# Selector decision (dev split v2, 88 images)

> **WITHDRAWN / NOT FROZEN (2026-10-29).** This decision was written before a root-cause audit of the review sheets
> and pipeline that the user requested after seeing apparent false positives. The config that had been written as
> `configs/selector_frozen.json` was moved to `configs/withdrawn/selector_frozen_WITHDRAWN_20261029.json`; no reference
> masks were generated from it. Nothing below is an accepted decision until the audit in `experiments/root_cause_audit.md`
> is complete. The experiment results themselves are preserved.

**Former decision (withdrawn): retain S1 (conservative gates) with the review flag as the frozen automatic selector.**
Previously frozen in `configs/selector_frozen.json` (content hash verified on load). S4's fallback is not used. No final ablation met the
acceptance criteria. Evidence labels are AI-assisted and unverified by a human; no pixel ground truth exists and no IoU is reported.

## What was tested (every row is in `experiments/selector_log.csv`; table in `experiments/selector_results.md`)

| config | change vs S1 | precision | recall | strict recall | false positives | missed | bad-boundary TPs | verdict |
|---|---|---|---|---|---|---|---|---|
| S1 | - | 0.990 | 0.876 | 0.876 | 1 | 14 | 0 | **frozen** |
| E1 alt shape gate (solidity >= 0.78, ellipse-IoU >= 0.72, but pred-IoU >= 0.95, stability >= 0.93, area >= 2%) | targets occluded eggs 11, 12, 144 | 0.766 | 0.956 | 0.920 | 33 | 5 | 4 | discard: +32 false positives in 22 images |
| E2 alt gate for mid-size candidates (area 2-10%, solidity >= 0.88, ellipse-IoU >= 0.89, pred-IoU >= 0.65, stability >= 0.55) | targets submerged eggs 124, 135, 145 | 0.931 | 0.956 | 0.929 | 8 | 5 | 3 | discard as automatic selector |
| E4 drop instances smaller than 25% of the largest | targets the earphone FP in image 16 | 1.000 | 0.876 | 0.876 | 0 | 14 | 0 | not adopted: single case |
| E5 E2 gate only when S1 selects nothing | follow-up from E2 review | 0.982 | 0.956 | 0.929 | 2 | 5 | 3 | not adopted automatically; suggestion tier |

Earlier ablations (S0, S2, S4) are in the same log. S2 and S4 were discarded on the previous review (S4's count accuracy was inflated by
cancelling errors).

## Why not E5 (the best trade-off)
Criteria were written before E5 ran: recall +5 points or more, precision loss 2 points or less, at most 5% of the new detections with bad boundaries.
Recall (+8.0) and precision (-0.8) passed; the boundary criterion failed: 3 of E5's 10 new detections are `egg_bad_boundary`
(117 ambiguous object cut by a rod, 123 partial egg behind plastic, 145 boxy left edge) and one new detection is a false positive (tray cavity in 42).
The E2/E5 gate parameters were chosen from the three submerged misses, so the dev gain is optimistic. E5's proposals are kept as a **suggestion tier**:
for images where S1 selects nothing, a reviewer may accept or reject them (see `data/review/dev_manual_selection_manifest.csv`).

## Findings by failure mode
- **Shape-rejected occluded eggs (11, 12, 144):** a proposal exists (box-IoU 0.83-0.95) but relaxing the shape gate admits many non-egg regions (E1). Not recoverable by a simple gate.
- **Confidence-rejected submerged eggs (124, 135, 145):** proposals exist; E2/E5 recover them with approximate boundaries (145 is bad-boundary).
- **False positives:** S1's only FP is the earphone bud (16); relaxed gates add keyboard keys (7 in E1), cloth, tile and tray cavities, foil, steel plate, wall regions.
- **Correct egg, poor boundary:** only seen in relaxed-gate additions (egg_bad_boundary 3-4 per relaxed config); S1's 99 detections have no bad-boundary verdict.

## Limits
Thresholds were not searched; gate values come from reviewed failures on the same 88 images, so dev numbers are optimistic. The eval split was not used.
