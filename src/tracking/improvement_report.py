"""Writes experiments/tracking_improvement_results.md from the two append-only logs (numbers are read, not retyped)."""
from __future__ import annotations

import csv
from collections import defaultdict

OLD, NEW = "experiments/video_tracking_log.csv", "experiments/video_tracking_improvement_log.csv"
NAMES = {"T0_gated": "T0", "T1_gated_plus_two_egg_prior": "T1", "T2_kalman": "T2", "T3_kalman_partial": "T3", "T3b_kalman_partial_bounded": "T3b"}


def main() -> None:
    rows = [r for r in csv.DictReader(open(OLD))] + [r for r in csv.DictReader(open(NEW))]
    main_rows = [r for r in rows if r["variant"] in NAMES and r.get("note", "") == ""]
    sens = [r for r in rows if "@" in r["variant"]]
    L = ["# Tracking improvement: measured comparison (T0, T1 unchanged; T2, T3, T3b new)", "",
         "Source logs: `experiments/video_tracking_log.csv` (T0, T1; untouched) and `experiments/video_tracking_improvement_log.csv` (new, append-only). Protocol and pre-registration: `experiments/tracking_improvement_audit.md`.",
         "Scoring is on COMPLETE detections only, against the unchanged single-annotator, AI-assisted proxy annotation (not human ground truth). Identity columns are whole-sequence; dev/held-out change only the frames used for coverage.", "",
         "| variant | subset | frames | expected | complete detections | coverage | missed | extra | ID switches | distinct IDs | extra IDs | gaps retained (Q480-558, Q648-750, P750-816) | detector ms | tracker ms |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in NAMES:
        for sub in ("dev", "heldout", "all"):
            for r in main_rows:
                if r["variant"] == v and r["subset"] == sub:
                    det = r.get("detections") or r.get("complete_detections")
                    dms = r.get("median_detect_ms_960w")
                    L.append(f"| {NAMES[v]} | {sub} | {r['frames_used']} | {r['expected_eggs']} | {det} | {r['coverage']} | {r['missed']} | {r['extra']} | {r['id_switches']} | {r['distinct_ids']} | {r['extra_ids']} | {r['gaps_retained']} | {dms} | {r['median_track_ms']} |")
    L += ["", "## Extra outputs of the new variants (all frames; not scored against the annotation)", "", "| variant | partial fragments assigned to a track | unassigned fragments | predicted-box rows | observed rows flagged uncertain |", "|---|---|---|---|---|"]
    for v in ("T2_kalman", "T3_kalman_partial", "T3b_kalman_partial_bounded"):
        for r in main_rows:
            if r["variant"] == v and r["subset"] == "all":
                L.append(f"| {NAMES[v]} | {r['partial_assigned']} | {r['partial_unassigned']} | {r['predicted_rows']} | {r['uncertain_rows']} |")
    L += ["", "## Sensitivity (one factor at a time, post-hoc robustness check, NOT used for selection)", ""]
    by = defaultdict(list)
    for r in sens:
        by[r["variant"].split("@")[0]].append(r)
    for v, rs in by.items():
        ok = [r for r in rs if r["id_switches"] == "0" and r["gaps_retained"] == "Q:1 Q:1 P:1"]
        bad = [f"{r['variant'].split('@')[1]} (ID switches {r['id_switches']}, gaps {r['gaps_retained']})" for r in rs if r not in ok]
        L.append(f"- {NAMES[v]}: {len(ok)} of {len(rs)} settings keep 0 ID switches and 3/3 retained gaps; failing: {', '.join(bad) if bad else 'none'}.")
    open("experiments/tracking_improvement_results.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
