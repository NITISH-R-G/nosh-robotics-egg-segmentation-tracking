# Review annotations (development split only)

Three annotation types are kept strictly separate. None of them is pixel-level ground truth.

| Type | File | What it says | Status |
|---|---|---|---|
| Image-level egg counts | `dev_egg_counts.csv`, `dev_egg_counts_corrections.csv` | How many eggs a reviewer sees in each dev image | Done (88 rows) |
| Candidate verdicts | `dev_instance_verdicts.csv` (planned) | Whether a specific cached SAM candidate mask is an egg | Not started |
| Pixel masks | none | Hand-drawn segmentation | Not produced |

All labels were produced by Claude (AI-assisted visual review of raw images, two passes on ambiguous
images) and have not been verified by a human. Evaluation-split images were never labelled.
Count corrections are recorded with a reason in `dev_egg_counts_corrections.csv`
(currently one: `not_damaged_117`, 2 -> 1, the lower tray shapes are empty depressions).
Validate with `pytest tests/test_review_labels.py`.
