"""Build report/REPORT.pdf from saved artifacts (numbers are read from the generated CSV/JSON files, not retyped where avoidable)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from PIL import Image as PILImage

SS = getSampleStyleSheet()
B = ParagraphStyle("b", parent=SS["BodyText"], fontSize=9, leading=12)
SM = ParagraphStyle("sm", parent=B, fontSize=7.5, leading=9.5)
H1, H2 = SS["Heading1"], SS["Heading2"]


def P(t, st=B):
    return Paragraph(t, st)


def img(path, w=17):
    im = PILImage.open(path)
    return Image(str(path), width=w * cm, height=w * cm * im.height / im.width)


def table(rows, widths=None, small=True):
    t = Table([[P(str(c), SM if small else B) for c in r] for r in rows], colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def exp_rows():
    d = pd.read_csv("experiments/classical_log.csv")
    rows = [["id", "what changed / why", "dev deltas vs comparison base (f1 vs R_rev, IoU vs R_rev, exact count)", "decision"]]
    for r in d[d.experiment_id.str.match(r"E\d\d")].itertuples():
        m = re.search(r'deltas vs comparison base: (\{.*?\})', str(r.decision))
        dl = json.loads(m.group(1)) if m else {}
        dec = str(r.decision).split(" (")[0]
        rows.append([r.experiment_id, r.motivation, ("%+.3f, %+.3f, %+.3f" % (dl["f1_rev"], dl["iou_rev"], dl["exact"])) if dl else "baseline", dec])
    return rows


def main():
    Path("report").mkdir(exist_ok=True)
    s1 = json.loads(Path("configs/frozen/S1_v1.json").read_text())
    cl = json.loads(Path("configs/frozen/classical_v1.json").read_text())
    ev = pd.read_csv("experiments/final_eval_with_counts.csv")
    s = []
    s += [P("Egg segmentation and tracking: report", H1),
          P("<b>Honesty statement.</b> No human pixel-level annotation exists. All 'ground truth' masks are SAM-derived pseudo-ground-truth (Task 1 output) and all egg counts, instance verdicts and "
            "video identity segments are AI-assisted (Claude Code visual review) and unverified by a human. Metrics are agreement with these references, not accuracy. Claude Code also wrote most "
            "of the code; see the AI disclosure at the end. Repository: https://github.com/NITISH-R-G/nosh-robotics-egg-segmentation-tracking . Tracking video (Google Drive): <b>link not created yet</b> "
            "(a footage-free schematic demo video of the selected T3b tracker is generated locally by <font face='Courier'>src/tracking/schematic_video_v2.py</font>; the original third-party footage is not redistributed).", B), Spacer(1, 6)]
    s += [P("1. Summary", H2),
          table([["", "dev (88 imgs)", "eval (22 imgs)"],
                 ["S1 (SAM + selector) exact egg count", "0.830 (73/88)", "0.864 (19/22)"],
                 ["classical exact egg count", "0.477 (42/88)", "0.545 (12/22)"],
                 ["classical instance F1 vs reviewed reference (dev)", "0.416", "not computed (no eval reference)"],
                 ["video (T3b, proxy): ID switches / gaps retained / complete-detection coverage", "0 / 3 of 3 / 0.963", "same clip, 1.000 on held-out frames"]], [7 * cm, 5 * cm, 5 * cm]),
          P("The pretrained route (S1) is much more reliable than the classical one on this dataset; the classical method is a documented weak baseline that fails whenever the background is as bright as the egg. "
            "The two methods are not ranked by pixel accuracy because no independent mask reference exists. S1 under-counts (0.159 of dev images, 0.136 of eval images) with one over-count in 88 dev images, and flags images where nothing passes its gates.", B)]
    s += [P("2. Data and leakage-aware split", H2),
          P("110 images (Roboflow egg export, CC BY 4.0). Exact (SHA) and near-duplicate pairs were linked into groups; groups are split 80:20 with seed 42, giving 88 dev / 22 eval images with no group crossing "
            "(<font face='Courier'>data/split_manifest.csv</font>). A first split (v1) was contaminated because pilot-inspected images ended in eval; v1 is kept for history and v2 forces those groups to dev. "
            "Constraints, grouping decisions and a leakage report are in <font face='Courier'>data/</font> and <font face='Courier'>src/data/splitting.py</font>; the evaluation images were not inspected or run until both methods were frozen.", B)]
    s += [P("3. Task 1: pretrained segmentation (S1)", H2),
          P("SAM ViT-B (official checkpoint, sha256 recorded in the frozen file) automatic mask generator, 32x32 points, run once for all 110 images; raw candidates are cached (14,400 candidates). "
            "A feature selector (area 1-60%, solidity >= 0.95, aspect <= 2, ellipse IoU >= 0.93, predicted IoU >= 0.9, stability >= 0.9, NMS) picks the egg masks; images with no selection are flagged for review, never guessed. "
            "A pre-registered set of selector experiments and a root-cause audit are in <font face='Courier'>experiments/</font>. An early freeze was withdrawn after display artefacts looked like false positives; the audit found a display defect, not a pipeline bug. "
            "A fallback using appearance descriptors and a box-prompted boundary refinement were tested on dev and rejected against pre-registered criteria. "
            "Outputs for all 110 images: <font face='Courier'>outputs/task1_sam_masks/</font> (15 are empty/flagged: 13 dev, 2 eval).", B)]
    s += [P("4. Task 2: classical segmentation and experiment log", H2),
          P(("Pre-registered 14 experiments (E00-E13), a greedy keep/discard rule evaluated on dev only, and leave-one-scene-out checks. Final config (frozen as classical_v1, config sha <font face='Courier'>%s</font>): "
            "Lab whiteness (L minus chroma), Otsu, 5 px open and close, hole fill, area 1-60%%, aspect <= 2.2, border-touching components >= 5%% of the image. Every trial, including failures, is in "
            "<font face='Courier'>experiments/classical_log.csv</font> with full parameters; the table gives the change, why and the keep/discard result (deltas of dev metrics against the comparison base).") % cl["config_sha"], B),
          Spacer(1, 4), table(exp_rows(), [1.2 * cm, 6 * cm, 6 * cm, 3.8 * cm]),
          P("Kept: E02, E09, E10, E11. Near misses decided by the rule: E08 (solidity) lost 0.030 macro IoU (limit 0.010); E12 (watershed) lost 0.057 exact-count rate (limit 0.030). Parameters of every trial: <font face='Courier'>experiments/classical_log.csv</font>.", B),
          img("outputs/figures/task3_dev_counts_and_experiments.png")]
    s += [PageBreak(), P("5. Task 3: evaluation and analysis (dev; Task 1 masks as reference)", H2),
          P("Reference: S1 raw masks (R_raw) and S1 plus 7 AI-assisted additions (R_rev), kept separate. Dev, 88 images, classical vs R_rev: instance P/R/F1 0.442/0.393/0.416, macro IoU 0.363, macro Dice 0.441, "
            "count exact 0.477 (over 0.170, under 0.352); vs R_raw macro IoU 0.366. 20 images have the right count with wrong instances (false positive standing in for a miss) against 0 for S1. "
            "S1's reference-based numbers are not accuracy (the reference is S1); its independent measure is verdict precision 0.990 / recall 0.876. Per scene the classical F1 is 0 on ash, striped cloth, egg tray, foil, tile and white-on-white, and 1.0 only on a dark keyboard background. "
            "Runtime: classical 0.05 s/image CPU; SAM candidates 16.4 s/image GPU plus selector 0.26 s.", B),
          img("outputs/figures/task3_dev_iou_scene_errors.png"),
          P("Qualitative: worst, best and category overlays (original | reference | S1 | classical). The failure is structural: brightness and low saturation describe the background as much as the egg.", B),
          img("outputs/classical/final_overlays/worst_0.jpg", 15), img("outputs/classical/final_overlays/category_0.jpg", 15),
          P("Failure types (details: <font face='Courier'>experiments/classical_failure_analysis.md</font>): empty prediction on white-on-white, tile, cloth and pale bowls; backlit egg darker than the wall; touching eggs merged; "
            "bright reflective foil or water beats the egg; ash, skin and rug regions merged with the egg. Improvement directions (not implemented): learned or colour-texture models, a background model, per-image adaptive features; none fits the classical constraint cheaply.", B)]
    s += [PageBreak(), P("6. Frozen methods and final evaluation (22 untouched images, one run)", H2),
          P("Both methods were frozen before the evaluation run with hashes, source-file hashes and dependency record (<font face='Courier'>configs/frozen/S1_v1.json</font> artifact sha <font face='Courier'>%s</font>, "
            "<font face='Courier'>classical_v1.json</font> <font face='Courier'>%s</font>); the run is recorded once in <font face='Courier'>outputs/eval/RUN_RECORD.json</font> and cannot be repeated. "
            "The protocol was written before freezing (<font face='Courier'>experiments/final_evaluation_protocol.md</font>). Eval counts were labelled by AI after the run from raw images but after the predicted counts had been printed (not blind). "
            "Note: <font face='Courier'>classical_log.csv</font> was appended with the freeze row after the freeze, so its current hash differs from the one recorded in classical_v1.json." % (s1["artifact_sha256"][:16], cl["artifact_sha256"][:16]), B),
          table([["method", "exact", "over", "under", "mean abs err", "predicted / expected eggs"],
                 ["S1", "0.864 (19/22)", "0.000", "0.136", "0.136", "22 / 25"],
                 ["classical", "0.545 (12/22)", "0.273", "0.182", "0.500", "26 / 25"]], [3 * cm, 3 * cm, 2 * cm, 2 * cm, 3 * cm, 4 * cm]),
          P("Classical vs S1 raw masks (agreement only, S1 non-empty on 20/22): macro IoU 0.315, Dice 0.439; instances at IoU 0.5: TP 6, FP 20, FN 16. S1 misses: 10 and 122 (flagged), 142 (one of three touching eggs). "
            "Classical: empty on 126 and 31; background regions added or merged on most others. Dev and eval are consistent in direction; 22 images allow no significance claim.", B),
          img("outputs/figures/task3_eval_counts.png"), img("outputs/eval/overlays/sheet_2.jpg", 14)]
    s += [PageBreak(), P("7. Task 4: egg tracking in the video", H2),
          P("Video Y4JyV94y2_A (AV1, 1920x1080, 24 fps); experiment interval frames 240-999 (title and explanation cards excluded). The frozen classical config also segments the wall, shirt and hand on this video, so a "
            "video-specific detector (grey >= 200, saturation <= 60, morphology, area/solidity/aspect/ellipse-fill filters; fixed a priori) is used. Sample: every 6th frame (4 Hz, 127 frames), alternate 1-second blocks as dev / held-out. "
            "Trackers: T0 (gated Hungarian), T1 (T0 + 'two eggs' prior) and, in a later improvement round, T2 (Kalman), T3 (Kalman + partial egg fragments) and <b>T3b (selected: T3 with prediction kept inside the frame)</b>. "
            "T0-T3 and their outputs are preserved unchanged.", B),
          table([["", "T0", "T1", "T2", "T3", "T3b (selected)"], ["distinct IDs for 2 eggs", "4", "3", "2", "2", "2"], ["ID switches", "3", "2", "0", "0", "0"],
                 ["gaps with identity retained (of 3)", "0", "1", "3", "3", "3"], ["complete-detection coverage dev / held-out", "0.963 / 1.000", "same", "same", "same", "same"],
                 ["detector + tracker ms/frame (CPU)", "8.3 + 0.115", "8.3 + 0.106", "6.1 + 0.18", "9.7 + 0.22", "9.8 + 0.21"]], [5.2 * cm, 2.4 * cm, 2.4 * cm, 2.2 * cm, 2.2 * cm, 3 * cm]),
          P("<b>What T3b adds.</b> The pick-up failure of T0/T1 was mainly a detector problem: eggs under the hand were present as white fragments but rejected by whole-egg shape filters, so P had no detection from frame 732 to 810 and Q's detection was given P's track. "
            "T3b accepts fragments (max 3 per frame, area 0.3-5% of the frame, solidity >= 0.65, aspect <= 3.5) as <i>partial</i> observations that can update a track but never start one, predicts missing tracks with a constant-velocity Kalman filter (predicted boxes are reported separately and are never counted as detections), "
            "gates the Hungarian assignment on Mahalanobis distance and IoU and flags ambiguous assignments. T2 (Kalman without fragments) reaches the same headline numbers only because the long-lost track's growing uncertainty makes it a cheaper match at frame 750 (ambiguous, fragile): not selected.", B),
          P("<b>Result and honest limits.</b> On this single 31-second clip T3b has <b>0 identity switches and 3 of 3 annotated occlusion gaps retained</b> (T1: 2 and 1 of 3), measured against a single-annotator, AI-assisted proxy annotation (<font face='Courier'>data/video/identity_segments.csv</font>), not human ground truth. "
            "<b>This is not independent generalisation validation:</b> T3b was designed after inspecting the critical pick-up sequence (frames 690-780), the held-out block 744-767 had been viewed by the annotator and the designer, the annotation was written after viewing every sampled frame, and T3b was added after T3 failed a pre-registered plausibility criterion. "
            "Metrics saturate (0 / 3 of 3), so they cannot rank T3b against other good trackers. Scoring uses complete detections only (fragments are shown separately); detection coverage is unchanged: 0.963 dev (4 misses: caption arrow at 336/342, hand at 732/738) and 1.000 held-out, 0 extra complete detections.", B),
          P("<b>Uncertainty is kept, not hidden.</b> T3b flags 4 observed associations as uncertain (all 'ambiguous association'): frame 540 (ID 1 fragment re-acquired after the hand carry), 684 (ID 1 fragment re-acquired), 750 (ID 2 fragment during the pick-up) and 810 (ID 2 complete detection after the carry). "
            "ID 2 has no observation from frame 762 to 804: T3b shows a predicted box for 3 samples (762-774), then nothing (identity kept, position unknown) until the re-appearance at 810; the earlier T3 variant predicted up to frame 792 and its predicted centre left the frame (y &lt; 0) at 780-792, which is why T3b bounds the prediction. "
            "Of 31 frames with fragments, all 21 fragments assigned to a track lie on an egg; 9 unassigned fragments are a false detection of the white 'Plane surface' caption arrow (frames 246-294), harmless because fragments cannot start tracks. Sensitivity (post-hoc, one factor at a time): T3b keeps 0 switches in 16 of 17 settings and fails at the lowest process noise (q_acc 0.1).", B),
          img("outputs/figures/video_tracking_T1_vs_T3b.png", 16),
          P("Figure: egg-box centre x per ID from the saved tracks (no third-party footage). Top T1: ID 2 jumps to the wrong egg at the pick-up (dotted line, about frame 747). Bottom T3b: dots = observed whole egg, triangles = fragments, hollow = predicted, red diamonds = uncertain association, grey crosses = false fragments; dashed lines = no observation between samples.", SM),
          P("Proposed demonstration (not uploaded): <font face='Courier'>egg_tracking_T3b_schematic.mp4</font>, a footage-free schematic rendering of the T3b tracks (solid = observed, dotted = fragment, dashed = predicted, '?' = uncertain, per-ID status; boxes update at 4 Hz and are held). The source footage (MocomiKids, standard YouTube licence) and the annotated original-footage video are not redistributed. "
            "Tracking video (Google Drive): <b>not uploaded; link not available at the time of this report build</b>. Replace this line with the verified public link before submission. Details: <font face='Courier'>experiments/tracking_improvement_audit.md</font>, <font face='Courier'>experiments/tracking_improvement_results.md</font>, <font face='Courier'>experiments/video_tracking_results.md</font>.", B)]
    s += [P("8. Limitations and open issues", H2),
          P("No human-verified labels; references derive from S1; 22 evaluation images; eval counts not blind; video evaluation is a proxy on one clip with 3 occlusion events; the Drive video link does not exist yet; "
            "the classical method is a weak baseline; S1 boundaries look smoother/larger than the visible egg edge (unmeasured). Four reviewed additions are first-pass only and seven development images have a bad-boundary S1 candidate excluded from the reviewed reference (one of them uncertain).", B),
          P("9. Reproduce", H2),
          P("See README.md (setup, data/weights locations, exact commands). Tests: 213 pass. Evaluation re-run is blocked by design.", B),
          P("10. AI-tool disclosure", H2),
          P("Claude Code (Anthropic) wrote most code, tests, protocols and analyses under the author's direction and produced the AI-assisted labels by viewing images; these labels are unverified by a human. "
            "The author is expected to explain the methods in the interview; withdrawn or rejected steps (premature selector freeze, contaminated split v1, rejected fallback and boundary refinement) are recorded in the logs.", B)]
    SimpleDocTemplate("report/REPORT.pdf", pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm, title="Egg segmentation and tracking report").build(s)
    print("report/REPORT.pdf written")


if __name__ == "__main__":
    main()
