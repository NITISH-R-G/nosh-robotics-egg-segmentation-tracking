# Selector results on dev (88 images, split v2)

Generated from `experiments/selector_log.csv` and `experiments/selector_instance_metrics.csv` by `python -m src.experiments.results_table`. Labels are AI-assisted image-level counts and AI-assisted manual candidate verdicts, not pixel ground truth. No IoU is reported.

| config | exact | over | under | mean abs err | median | max | precision (lenient) | recall (lenient) | precision (strict) | recall (strict) | F1 (lenient) | bad-boundary TPs | FP | missed | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0_pilot_filter | 0.409 | 0.557 | 0.034 | 1.477 | 1.0 | 14 | 0.447 | 0.938 | 0.435 | 0.912 | 0.606 | 3.0 | 131.0 | 7.0 | 27.53 |
| S1_conservative | 0.83 | 0.011 | 0.159 | 0.17 | 0.0 | 1 | 0.99 | 0.876 | 0.99 | 0.876 | 0.93 | 0.0 | 1.0 | 14.0 | 23.06 |
| S2_conservative_border_rule | 0.795 | 0.057 | 0.148 | 0.205 | 0.0 | 1 | 0.943 | 0.876 | 0.943 | 0.876 | 0.908 | 0.0 | 6.0 | 14.0 | 22.71 |
| S3_S1_plus_review_flag | 0.83 | 0.011 | 0.159 | 0.17 | 0.0 | 1 | 0.99 | 0.876 | 0.99 | 0.876 | 0.93 | 0.0 | 1.0 | 14.0 | 21.77 |
| S4_S1_plus_relaxed_fallback | 0.955 | 0.011 | 0.034 | 0.045 | 0.0 | 1 | 0.964 | 0.947 | 0.928 | 0.912 | 0.955 | 4.0 | 4.0 | 6.0 | 22.31 |
| E1_S1_alt_shape | 0.705 | 0.25 | 0.045 | 0.409 | 0.0 | 6 | 0.766 | 0.956 | 0.738 | 0.92 | 0.85 | 4.0 | 33.0 | 5.0 | 21.09 |
| E2_S1_alt_conf | 0.875 | 0.08 | 0.045 | 0.125 | 0.0 | 1 | 0.931 | 0.956 | 0.905 | 0.929 | 0.943 | 3.0 | 8.0 | 5.0 | 20.96 |
| E4_S1_rel_area | 0.841 | 0.0 | 0.159 | 0.159 | 0.0 | 1 | 1.0 | 0.876 | 1.0 | 0.876 | 0.934 | 0.0 | 0.0 | 14.0 | 21.59 |
| E5_S1_alt_gate_fallback | 0.943 | 0.011 | 0.045 | 0.057 | 0.0 | 1 | 0.982 | 0.956 | 0.955 | 0.929 | 0.969 | 3.0 | 2.0 | 5.0 | 24.8 |

## Exact-count rate by difficulty tag (n images)

| tag | S0_pilot_filter | S1_conservative | S2_conservative_border_rule | S3_S1_plus_review_flag | S4_S1_plus_relaxed_fallback | E1_S1_alt_shape | E2_S1_alt_conf | E4_S1_rel_area | E5_S1_alt_gate_fallback |
|---|---|---|---|---|---|---|---|---|---|
| backlit_silhouette (1) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| border_clipped (6) | 0.333 | 1.0 | 1.0 | 1.0 | 1.0 | 0.667 | 0.833 | 1.0 | 1.0 |
| hand_held (7) | 0.429 | 0.857 | 0.857 | 0.857 | 1.0 | 0.857 | 0.857 | 0.857 | 1.0 |
| in_tray_uncertain (3) | 0.333 | 0.0 | 0.333 | 0.0 | 1.0 | 0.333 | 1.0 | 0.0 | 1.0 |
| leaf_occluded (3) | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.333 | 0.0 | 0.333 |
| small_egg (4) | 0.25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| submerged (3) | 0.333 | 0.0 | 0.0 | 0.0 | 0.333 | 0.0 | 1.0 | 0.0 | 1.0 |
| touching_or_adjacent (5) | 0.4 | 0.8 | 0.6 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| translucent_on_bag (1) | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| untagged (55) | 0.455 | 0.945 | 0.891 | 0.945 | 0.982 | 0.764 | 0.909 | 0.964 | 0.982 |
