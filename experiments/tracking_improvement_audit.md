# Tracking improvement audit (written BEFORE any new tracker code)

Scope: Task 4 only. T0/T1 source, configs, logs and outputs stay unchanged; new variants get new files. The identity annotation (`data/video/identity_segments.csv`, `ambiguous_frames.txt`) is single-annotator, AI-assisted and is not changed. Nothing here is committed or pushed.

## 1. Exact baseline (saved artifacts: `experiments/video_tracking_log.csv`, `outputs/video/tracking_*/tracks.csv`)
| variant | subset | frames | expected | detections | coverage | missed | extra | ID switches | distinct IDs | gaps retained (Q480-558, Q648-750, P750-816) | detector ms (960 px) | tracker ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | dev | 63 | 107 | 103 | 0.9626 | 4 | 0 | 3 | 4 | 0, 0, 0 | 8.3 | 0.115 |
| T0 | held-out | 58 | 98 | 98 | 1.0 | 0 | 0 | 3 | 4 | 0, 0, 0 | 8.3 | 0.115 |
| T0 | all | 121 | 205 | 201 | 0.9805 | 4 | 0 | 3 | 4 | 0, 0, 0 | 8.3 | 0.115 |
| T1 | dev | 63 | 107 | 103 | 0.9626 | 4 | 0 | 2 | 3 | 1, 0, 0 | 8.3 | 0.106 |
| T1 | held-out | 58 | 98 | 98 | 1.0 | 0 | 0 | 2 | 3 | 1, 0, 0 | 8.3 | 0.106 |
| T1 | all | 121 | 205 | 201 | 0.9805 | 4 | 0 | 2 | 3 | 1, 0, 0 | 8.3 | 0.106 |
(The identity columns are computed over the whole sequence and are identical for every subset because the evaluation scores the same ID sequences; the dev/held-out split only changes which frames contribute to coverage.)
Dev/held-out blocks are 1-second blocks of 24 frames starting at 240 (`is_dev`: `(frame//24)` even). The critical pick-up frames 744-767 are in a HELD-OUT block (block 31); 720-743 and 768-791 are dev blocks. The pick-up sequence was nevertheless viewed in this audit (and earlier) by the annotator, so held-out frames are not unseen; the rule below is how selection stays honest.

## 2. The sequence around source frames 690-816 (T0 and T1 produce identical tracks here)
- Frames 690-726: one track, ID 2, on egg P at x ~ 473 px (box width 226-242), steady. The other egg Q is at the far left under the hand and is **not output**.
- Frames 732, 738, 744: **no detection at all**.
- Frame 750 onward: the only detection is at x ~ 204-216 px, y ~ 580 (egg Q, stationary at the far left) and it is assigned to **ID 2**, the track of P. P (picked up and carried) is not output until frame 810, when a detection at x ~ 1140 appears and gets a new ID (ID 4 in T0, ID 3 in T1).
- Q had its own track (ID 1 in both variants): seen at x ~ 205 until frame 504, re-linked after the carry to the right (T1 only), last seen at x ~ 1285 at frame 642, then carried back to the far left by the hand.
- Q's resting place at the far left (x ~ 205-215) is the same as its position at the start of the video; P's last observation before the pick-up was x ~ 481 at frame 726.

## 3. Failure classification (evidence from the detector's component dump, frames 690-756)
Pixel-level: the eggs are **not invisible**; the white-pixel mask contains egg fragments in nearly every frame, but the shape filters reject them (area, aspect, solidity, ellipse fill; all values in `src/tracking/detect.py`):
| frame | components found (x, y, area at 1920 px scale, solidity, aspect, ellipse fill) -> verdict |
|---|---|
| 690 | (288, 493, 9028, 0.98, 2.08, 0.87) rejected by aspect (Q partly under the hand, ~25% of an egg area); (472, 655, 39752) kept (P) |
| 696 / 708 / 720 | Q fragments (307, 548) / (301, 567) / (298, 567), areas 7016 / 7500 / 9472, aspect 1.88-2.25, rejected; P kept |
| 726 | P kept (486, 658, 34476); Q fragment (282, 567, 3376) rejected by area and aspect |
| 732 | only (544, 668, 12508, aspect 1.96, fill 0.85) rejected by aspect and fill: this is **P partly under the fingers** |
| 738 | P fragment (566, 670, 6660, aspect 3.15) and Q fragment (116, 573, 4436, solidity 0.81) rejected |
| 744 | P (592, 638, 10168, aspect 2.37) and Q (120, 623, 2740) fragments rejected |
| 750 | **P in the hand, blurred**: (693, 458, 8936, aspect 2.2, fill 0.83) rejected by aspect and fill; Q complete (205, 579, 33212) kept |
| 756 | P fragment (840, 341, 8440, solidity 0.69, fill 0.56) rejected; Q kept |
Conclusions:
1. **Primary failure = detector rejection of partially occluded eggs (strict whole-egg shape filters).** The pick-up is therefore an observation gap for P of about 13 sampled steps (732-810, apart from rejected fragments), during which the tracker sees only Q.
2. **Contributing failure = incorrect assignment.** With P unobserved, the Hungarian step gives Q's detection (277 px from P's last position, inside P's gate of (1.2 + 0.35*4) mean box diameters) to ID 2. The same cost model prefers ID 2 over the lost ID 1 (last seen about 1,080 px away) because the cost is normalised by a gate that grows with time. That is a wrong association, not a termination: no track expired here (max_age 24 steps).
3. **Track termination** is not the cause at this event (both tracks alive); it explains the T0 fragmentation at the earlier carry (ID 1 -> ID 3), which T1's two-egg prior already fixes.
4. **While the hand interacts, the eggs remain partially detectable as white fragments** (areas 20-35% of a whole egg), but no rule currently uses them.
Because P is undetected for the ~14 steps before Q's detection arrives, a better motion model or re-identification alone cannot fix this event (nothing to predict from); the hypothesis to test is that **accepting partial detections** is the smallest change that gives the tracker the evidence it needs. Mere track persistence/Kalman prediction is evaluated too, as the user's Phase 2 A/B, to document that it does not help here.

## 4. Appearance as a cue (checked on the saved boxes only)
Mean box width/height (px) by phase: Q at rest 252x200 (sd 3, 2), P at rest 243-241 x 209-207 (sd 1-3). Heights differ by about 9 px (4%) and widths by about 10 px, but the same egg changes with depth: the egg at the far left after frame 756 measures 244x190. Box size is therefore not a reliable discriminator; a colour/texture descriptor was not implemented because both eggs are visually identical white ellipses (hand-held/blurred eggs would add noise). Appearance is NOT used.

## 5. Pre-registered variants and rules (fixed before running; no changes after seeing results without a new variant name)
Detector D1 ("partial-tolerant", new file): everything in the current detector, plus up to 3 `partial` detections per frame: white components (same grey/saturation thresholds) not overlapping a complete detection with area 0.3%-5% of the 960 px frame, solidity >= 0.65, aspect <= 3.5. Partial detections can be associated to existing tracks but cannot start a track, and are flagged in the CSV.
Tracker K ("Kalman"): constant-velocity filter on (cx, cy) with damped velocity, observed vs predicted state flags, Mahalanobis + IoU gated Hungarian, "uncertain" flag when the best and second-best costs of a track or detection are within 20% of each other, the T1 two-egg relink prior retained, max_age 24 steps; predicted boxes are written to a separate column set and never counted as detections.
Variants: **T2_kalman** (D0 detector = current, tracker K) ; **T1P_partial** (T1 tracker, detector D1) ; **T3_kalman_partial** (tracker K, detector D1). Reference: T0, T1 unchanged.
Scoring (annotation unchanged): coverage/missed/extra, ID switches, distinct IDs, extra IDs and gap retention are computed on **complete** detections only, exactly as for T0/T1, because the existing identity rules (left/right order or x < threshold) were written for whole eggs and would mislabel fragments (a Q fragment at x = 300 px is above the x < 0.15 threshold). Partial detections are reported separately (counts per frame, plus a visual false-detection review of every partial detection in the dev frames and, for reporting, in the held-out frames). The critical sequence is reviewed qualitatively. Runtime per frame is measured for detector and tracker.
Acceptance (all required): fewer identity switches than T1; coverage maintained or improved; more retained occlusion gaps than T1 (1 of 3); no persistent false tracks and no implausible predicted boxes (visual and numeric: predicted boxes inside the frame, speed bounded); runtime acceptable (< 30 ms/frame for detector+tracker on CPU at 960 px); all tests pass. Otherwise T1 stays and the failure is documented.
Selection discipline: design choices were made from the component dump above (dev and held-out frames mixed, annotator saw all frames), so the selected variant is the product of looking at the event, not a blind test; with one pick-up event and three occlusion gaps in a single 31-second clip no generalisation claim is possible, and the report will say so.

---
## 6. Outcome, deviations and decision (appended after the runs; sections 1-5 above were not edited except two wording/number fixes made before any run: "about 13 sampled steps" and "about 1,080 px")
### Deviations from the pre-registration (all recorded in `experiments/video_tracking_improvement_log.csv`)
1. **T1P_partial (T1 tracker + partial detector) was NOT run.** The old tracker has no notion of detection kind and would have let fragments start tracks, which the pre-registration forbids. T2 (Kalman only) and T3 (Kalman + fragments) isolate the two ingredients instead.
2. **Sensitivity runs were added after seeing T2/T3** (one factor at a time, 17 settings per variant, logged with the note "sensitivity ... not used for selection") because T2's success looked fragile.
3. **T3b was added after T3 failed a pre-registered criterion** ("no implausible predicted boxes"): T3's predicted centre for ID 2 left the frame (y < 0) at frames 780, 786, 792 (3 of 21 predicted rows; 4 boxes partly outside). T3b changes only the bounded prediction (box kept fully inside 1920x1080, velocity zeroed on a clamped axis) and a 3-sample display budget. This is a post-result modification made on the same clip; it is disclosed as such and not counted as an independent confirmation. A byte-for-byte regression check confirmed that adding the option did not change T2/T3 outputs.

### What each variant showed (evidence in the log, `outputs/video/tracking_T*/tracks.csv`, `outputs/figures/video_tracking_T1_vs_T3b.png`)
- **T2 (Kalman prediction, persistence, observed/predicted flags, complete detections only): 0 ID switches, 2 IDs, 3/3 gaps retained, but for a weak reason.** At frame 750 the Q detection is assigned to the long-lost ID 1 because its Mahalanobis cost is lower: ID 1 has a position uncertainty of about 1,747 px (17 missed steps) so a 1,149 px offset is 0.66 sigma (d2 = 0.433), while ID 2 (3 missed steps, sigma about 365 px) would need a 336 px move, 0.92 sigma (d2 = 0.848). Growing uncertainty makes a lost track a more attractive match, an artefact of the cost normalisation, not evidence about which egg it is. The tracker itself flags the association "ambiguous". The result also fails when velocity damping is 0.5 (sensitivity). **T2 is not selected** even though its headline numbers equal T3b's.
- **T3 / T3b (Kalman + partial fragments): the same headline numbers, with evidence behind them.** 21 fragments were assigned to tracks. Reviewing the frames (sheet of all 31 frames with fragments): every one of the 21 assigned fragments lies on an egg: Q under the fingers at 480-498 and 684-720, Q carried in the hand at 540-552, P under the fingers at 732-756 (including P carried in the hand at 750-756), and Q partly merged with the "Raw Egg" caption arrow at 336, 342 (box too tall). The 10 unassigned fragments: 9 are the white caption arrow next to "Plane surface" (frames 246-294, **false detections**, harmless because they cannot start or move a track) and 1 is a blurred egg in the hand at 510 (true egg, unassigned). So with fragments both identities stay observed (through partial evidence) across the three annotated gaps; association is flagged uncertain at 4 observed rows, all with reason "ambiguous association": frames 540 (ID 1 fragment after the carry), 684 (ID 1 fragment), 750 (ID 2 fragment) and 810 (ID 2 complete detection after the carry). [Correction made before commit: an earlier draft wrongly said the 810 association was a two-egg-prior relink; the CSV shows an ordinary ambiguous association.]
- **Remaining weakness:** ID 2 (P) has no observation from frame 762 to 804. T3 predicted a box up to 792 (its centre left the frame at 780-792); T3b shows a predicted box only at 762, 768 and 774 and nothing at 780-804 (identity kept, position unknown). P's re-appearance at 810 is matched to ID 2 with an "ambiguous association" flag. q_acc = 0.1 (the lowest tested process noise) breaks T3/T3b (2 ID switches), so the fragment re-acquisition at 684 depends on a sufficiently large uncertainty.

### Acceptance (T3b), against the criteria in the task
| criterion | result |
|---|---|
| fewer identity switches than T1 | PASS: 0 vs 2 (dev, held-out and all) |
| detection coverage improved or maintained | PASS: complete-detection coverage identical (0.9626 dev, 1.0 held-out); fragments add observations but are not scored |
| gap retention improved | PASS: 3 of 3 vs 1 of 3 |
| no persistent false tracks / implausible predicted boxes | PASS with notes: 2 IDs only; 9 false unassigned fragments (caption arrow) that never become tracks; all 12 predicted rows are fully inside the frame (T3 failed this, hence T3b); predictions are extrapolations limited to 3 samples (0.75 s) and the ID 2 prediction at 762-768 is clamped against the top frame edge |
| acceptable runtime | PASS: detector 9.76 ms + tracker 0.211 ms per frame (CPU, 960 px, from cached frames; the T0/T1 figure of 8.3 ms was measured while decoding the video, so the two are not strictly comparable) |
| tests | PASS: 213 passed (197 existing + 16 new in `tests/test_tracking_v2.py`) |
**Decision (pending your approval):** T3b meets the pre-registered criteria and is the smallest change with evidence behind it; T2 is rejected for the reason above; T1 stays the published baseline until you approve replacing it. **Is it genuinely better?** On this clip, yes: the ID error at the pick-up is gone and, unlike T2, it is supported by observed fragments of both eggs. The proxy numbers saturate (0 switches, 3 of 3 gaps), so they cannot rank T3b against any other equally good tracker, and they come from one 31-second clip with three annotated gaps, one annotator, annotation written after viewing every frame, and a design chosen after viewing the critical event (held-out frames 744-767 included). No generalisation claim is justified.
