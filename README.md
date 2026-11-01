# Egg segmentation and tracking (CV intern task)

Pretrained SAM segmentation of 110 egg images, a classical image-processing segmenter, an evaluation, and egg tracking in one video.
**Read `report/REPORT.pdf` first** (it summarises everything below). All reference masks are SAM-derived pseudo-ground-truth and all counts/verdicts are AI-assisted and unverified by a human; no pixel-level human annotation exists.

## Status (honest)
| Task | State |
|---|---|
| 1. SAM masks for all 110 images | done: `outputs/task1_sam_masks/` (S1, 110 masks; 15 are empty because no candidate passed the conservative gates, flagged for review) |
| 2. 80:20 leakage-aware split, classical method, experiment log | done: `data/split_manifest.csv` (88 dev / 22 eval), `experiments/classical_log.csv`, `configs/frozen/classical_v1.json` |
| 3. Evaluation and analysis | done: `experiments/classical_results.md`, `experiments/classical_failure_analysis.md`, `experiments/final_evaluation_results.md`, `outputs/figures/` |
| 4. Video tracking | done with a proxy-evaluated tracker, **T3b (selected)**: `experiments/video_tracking_results.md`, `experiments/tracking_improvement_audit.md`, `experiments/tracking_improvement_results.md`; footage-free schematic demo video of T3b (local, `src/tracking/schematic_video_v2.py`); **Google Drive link: not created yet** |

Headline numbers (egg-count agreement with AI-assisted counts; not pixel accuracy): S1 0.830 dev / 0.864 eval exact; classical 0.477 dev / 0.545 eval. The classical method is a documented weak baseline (instance F1 0.42 on dev). Video (single 31-second clip, single-annotator AI-assisted proxy annotation, NOT human ground truth and NOT independent generalisation validation): the selected tracker T3b has 0 identity switches and 3 of 3 annotated occlusion gaps retained, versus T1's 2 switches and 1 of 3; complete-detection coverage is unchanged (0.963 dev / 1.000 held-out). T3b was designed after inspecting the critical pick-up sequence and the held-out block was viewed; 4 associations are flagged uncertain and one egg has no observation for frames 762-804 (see the results files).

## Setup (Windows, Python 3.12; SAM and the torch CUDA wheels were used with an NVIDIA GPU)
```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock.txt
```
`requirements.lock.txt` pins every package (including `reportlab`, used only by `src/report/build_report.py`) and contains the PyTorch CUDA 12.8 wheel index line it needs for `torch==2.11.0+cu128`. Not in the lock file: `yt-dlp` (2026.10.26) and `yt-dlp-ejs` (0.8.0), needed only to re-download the video.
Data and weights are not in the repository:
- Dataset: place the Roboflow export (`train/ valid/ test/` folders) in `data_raw/` (zip link from the task PDF; license CC BY 4.0).
- SAM ViT-B checkpoint: `models/sam_vit_b_01ec64.pth` (official Meta `sam_vit_b_01ec64.pth`, sha256 `ec2df627...` recorded in `outputs/sam_candidates/run.json` after the cache run).
- Video: download YouTube `Y4JyV94y2_A` with `yt-dlp --js-runtimes node` (needs Node and yt-dlp-ejs); the analysis reads `scratch/video_dl/egg_video.f399.mp4` (sha256 `841d50d8...`). Only the AV1 video stream is used.

## Run
Everything is run from the repository root with `PYTHONPATH=.`.
```bash
# tests (213 expected to pass)
.venv/Scripts/python -m pytest -q

# inventory and the group-aware split (the committed data/split_manifest.csv is the one used; reproduction by re-running was not re-verified in the release audit)
.venv/Scripts/python -m src.data.inventory --root data_raw
.venv/Scripts/python -m src.data.splitting
# Task 1: cache raw SAM candidates for all images (about 30 minutes on a GPU), audit, then selector experiments (dev only)
.venv/Scripts/python -m src.segmentation.cache_candidates
.venv/Scripts/python -m src.segmentation.audit_cache
.venv/Scripts/python -m src.experiments.selector_experiments
# Task 2: classical experiments (14 pre-registered experiments, dev only, about 5 minutes) and report
.venv/Scripts/python -m src.experiments.classical_experiments
.venv/Scripts/python -m src.experiments.classical_report
# Freeze both methods, then ONE evaluation run (refuses to run twice) and its report
.venv/Scripts/python -m src.experiments.freeze_methods
.venv/Scripts/python -m src.experiments.final_eval
.venv/Scripts/python -m src.experiments.final_report
.venv/Scripts/python -m src.analysis.task3_plots
# Task 4 video: T0/T1 (first run decodes the video, slow); T2/T3/T3b improvement round (cached 960 px frames, built on first use); comparison figure, results table and the footage-free schematic video
.venv/Scripts/python -m src.tracking.video_experiment
.venv/Scripts/python -m src.tracking.improvement_experiment
.venv/Scripts/python -m src.tracking.improvement_report
.venv/Scripts/python -m src.tracking.trajectory_plot_v2
.venv/Scripts/python -m src.tracking.schematic_video_v2 outputs/video/tracking_T3b_kalman_partial_bounded/tracks.csv outputs/video/final/egg_tracking_T3b_schematic.mp4 "T3b tracker output (schematic)"
```
Some entry points take command-line options (see `--help`). Re-running `final_eval` after the first run is blocked by `outputs/eval/RUN_RECORD.json` on purpose.

## Layout
- `src/data`, `src/segmentation`, `src/evaluation`, `src/experiments`, `src/analysis`, `src/tracking`; `tests/` (213 tests).
- `experiments/`: protocols (written before runs), append-only logs (`selector_log.csv`, `classical_log.csv`, `video_tracking_log.csv`, `video_tracking_improvement_log.csv`, `protocol_amendment_log.csv`), generated results and failure analyses.
- `configs/frozen/`: S1 and classical configs with hashes, source-file hashes and dependency record; `configs/withdrawn/` keeps a withdrawn early freeze for the record.
- `data/`: split manifest (v2; the contaminated v1 is kept for history), grouping decisions, AI-assisted review labels (`data/review/`), video identity segments (`data/video/`).

## Methods in one paragraph each
- **S1 (Task 1):** SAM automatic mask generator candidates (cached for every image) are filtered by area, solidity, aspect ratio, ellipse fit, predicted IoU and stability, then non-maximum suppression. Images where nothing passes are flagged, not guessed. It misses some eggs (about 16% of dev counts are under-counts), has one over-count in 88 dev images (verdict precision 0.990), and masks look smoother than the true edge.
- **Classical (Task 2):** Lab whiteness (L minus chroma), Otsu, morphology, hole fill, shape/area/border filters; selected by a pre-registered greedy rule over 14 experiments on dev only. It fails whenever the background is as bright as the egg.
- **Video (Task 4):** a video-specific fixed white-threshold detector plus, for the selected T3b tracker, partial egg-fragment detections (fragments can update but never start a track), a constant-velocity Kalman filter with predicted boxes reported separately from observed ones, gated Hungarian association and explicit uncertainty flags. Earlier variants T0 (gated Hungarian), T1 (+ two-egg prior) and the intermediate T2/T3 are kept unchanged for comparison. The image-set classical config was not usable on the video (it also segments the wall, shirt and hand).

## Dataset attribution and licences
- Egg images: "egg" dataset by the Roboflow Universe user fire-bf7qe, https://universe.roboflow.com/fire-bf7qe/egg-vzhtr, licensed **CC BY 4.0**. The dataset is not redistributed here (download it yourself), but overlay sheets and example figures contain individual images from it (with contours drawn on them) and the masks are derived from them; they are shared under the same CC BY 4.0 attribution, with changes made by this project.
- Segment Anything (Meta AI), Apache-2.0; the ViT-B checkpoint is downloaded separately.
- Video: the YouTube clip Y4JyV94y2_A (MocomiKids, standard YouTube licence) is third-party content. Neither the clip nor any frame of it is included in this repository, and the annotated footage video is not published. The report shows a trajectory comparison figure generated from our own tracking CSVs instead of stills; the proposed demonstration is a footage-free schematic video of the T3b tracks (`python -m src.tracking.schematic_video_v2 outputs/video/tracking_T3b_kalman_partial_bounded/tracks.csv <out.mp4> <label>`); it has not been uploaded.
- Original project code is released under the MIT License (`LICENSE`). The MIT licence does not cover the dataset (CC BY 4.0), SAM (Apache-2.0) or the third-party video.

## Split and leakage
80:20 by groups (exact-duplicate and near-duplicate images linked, seed 42, 88 dev / 22 eval). The first split (v1) was contaminated by pilot inspection and is preserved but not used; v2 forced pilot-inspected groups to dev. See `experiments/root_cause_audit.md` and the split files for what happened and why.

## AI-tool disclosure
Claude Code (Anthropic) wrote most of the code, tests, protocols and analysis documents under the author's direction, and produced the AI-assisted labels (egg counts, instance verdicts, scene categories, boundary ratings, video identity segments) by viewing images. Those labels are **not** human-verified and are named AI-assisted wherever they are used. The author must be able to explain the methods; the interview walk-through should not rely on this README alone. Metrics, thresholds and decisions are recorded in the logs; where a step failed or was withdrawn (premature selector freeze, contaminated split v1, rejected fallback and refinement experiments) the record says so.

## Known limitations
No human ground truth; references derive from S1; 22 evaluation images; eval counts were labelled after seeing predicted counts; video evaluation is a single-annotator proxy on one clip (T3b designed after inspecting the critical sequence; held-out frames were viewed; not independent validation); the Google Drive video link does not exist yet.
