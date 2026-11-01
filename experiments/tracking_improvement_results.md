# Tracking improvement: measured comparison (T0, T1 unchanged; T2, T3, T3b new)

Source logs: `experiments/video_tracking_log.csv` (T0, T1; untouched) and `experiments/video_tracking_improvement_log.csv` (new, append-only). Protocol and pre-registration: `experiments/tracking_improvement_audit.md`.
Scoring is on COMPLETE detections only, against the unchanged single-annotator, AI-assisted proxy annotation (not human ground truth). Identity columns are whole-sequence; dev/held-out change only the frames used for coverage.

| variant | subset | frames | expected | complete detections | coverage | missed | extra | ID switches | distinct IDs | extra IDs | gaps retained (Q480-558, Q648-750, P750-816) | detector ms | tracker ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | dev | 63 | 107 | 103 | 0.9626 | 4 | 0 | 3 | 4 | 2 | Q:0 Q:0 P:0 | 8.3 | 0.115 |
| T0 | heldout | 58 | 98 | 98 | 1.0 | 0 | 0 | 3 | 4 | 2 | Q:0 Q:0 P:0 | 8.3 | 0.115 |
| T0 | all | 121 | 205 | 201 | 0.9805 | 4 | 0 | 3 | 4 | 2 | Q:0 Q:0 P:0 | 8.3 | 0.115 |
| T1 | dev | 63 | 107 | 103 | 0.9626 | 4 | 0 | 2 | 3 | 1 | Q:1 Q:0 P:0 | 8.3 | 0.106 |
| T1 | heldout | 58 | 98 | 98 | 1.0 | 0 | 0 | 2 | 3 | 1 | Q:1 Q:0 P:0 | 8.3 | 0.106 |
| T1 | all | 121 | 205 | 201 | 0.9805 | 4 | 0 | 2 | 3 | 1 | Q:1 Q:0 P:0 | 8.3 | 0.106 |
| T2 | dev | 63 | 107 | 103 | 0.9626 | 4 | 0 | 0 | 2 | 0 | Q:1 Q:1 P:1 | 6.08 | 0.183 |
| T2 | heldout | 58 | 98 | 98 | 1.0 | 0 | 0 | 0 | 2 | 0 | Q:1 Q:1 P:1 | 6.08 | 0.183 |
| T2 | all | 121 | 205 | 201 | 0.9805 | 4 | 0 | 0 | 2 | 0 | Q:1 Q:1 P:1 | 6.08 | 0.183 |
| T3 | dev | 63 | 107 | 103 | 0.9626 | 4 | 0 | 0 | 2 | 0 | Q:1 Q:1 P:1 | 9.69 | 0.221 |
| T3 | heldout | 58 | 98 | 98 | 1.0 | 0 | 0 | 0 | 2 | 0 | Q:1 Q:1 P:1 | 9.69 | 0.221 |
| T3 | all | 121 | 205 | 201 | 0.9805 | 4 | 0 | 0 | 2 | 0 | Q:1 Q:1 P:1 | 9.69 | 0.221 |
| T3b | dev | 63 | 107 | 103 | 0.9626 | 4 | 0 | 0 | 2 | 0 | Q:1 Q:1 P:1 | 9.76 | 0.211 |
| T3b | heldout | 58 | 98 | 98 | 1.0 | 0 | 0 | 0 | 2 | 0 | Q:1 Q:1 P:1 | 9.76 | 0.211 |
| T3b | all | 121 | 205 | 201 | 0.9805 | 4 | 0 | 0 | 2 | 0 | Q:1 Q:1 P:1 | 9.76 | 0.211 |

## Extra outputs of the new variants (all frames; not scored against the annotation)

| variant | partial fragments assigned to a track | unassigned fragments | predicted-box rows | observed rows flagged uncertain |
|---|---|---|---|---|
| T2 | 0 | 0 | 24 | 4 |
| T3 | 21 | 10 | 21 | 4 |
| T3b | 21 | 10 | 12 | 4 |

## Sensitivity (one factor at a time, post-hoc robustness check, NOT used for selection)

- T2: 16 of 17 settings keep 0 ID switches and 3/3 retained gaps; failing: vel_damp=0.5 (ID switches 2, gaps Q:1 Q:0 P:0).
- T3: 16 of 17 settings keep 0 ID switches and 3/3 retained gaps; failing: q_acc=0.1 (ID switches 2, gaps Q:1 Q:0 P:1).
- T3b: 16 of 17 settings keep 0 ID switches and 3/3 retained gaps; failing: q_acc=0.1 (ID switches 2, gaps Q:1 Q:0 P:1).
