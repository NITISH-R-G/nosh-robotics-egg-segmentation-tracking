"""Read-only cross-artifact consistency checks; prints one line per check and writes experiments/consistency_checks.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

import fitz
import pandas as pd

R = []


def chk(name, ok, evidence, detail=""):
    R.append({"check": name, "result": "PASS" if ok else "FAIL", "evidence": evidence, "detail": detail})


def main():
    sp = pd.read_csv("data/split_manifest.csv")
    chk("split 88 dev / 22 eval, 110 total", sp.split.value_counts().to_dict() == {"dev": 88, "eval": 22} and len(sp) == 110, "data/split_manifest.csv", str(sp.split.value_counts().to_dict()))
    ids = pd.read_csv("outputs/eval/eval_split.csv").image_id
    chk("eval run used exactly the 22 eval ids", set(ids) == set(sp[sp.split == "eval"].image_id) and len(ids) == 22, "outputs/eval/eval_split.csv")
    for sub in ("S1", "classical"):
        have = {p.stem for p in Path(f"outputs/eval/{sub}/masks").glob("*.png")}
        chk(f"eval masks {sub} = 22 eval ids", have == set(ids), f"outputs/eval/{sub}/masks", f"{len(have)} files")
    rec = json.loads(Path("outputs/eval/RUN_RECORD.json").read_text())
    s1 = json.loads(Path("configs/frozen/S1_v1.json").read_text()); cl = json.loads(Path("configs/frozen/classical_v1.json").read_text())
    chk("RUN_RECORD hashes equal frozen artifact hashes", rec["s1_artifact_sha256"] == s1["artifact_sha256"] and rec["classical_artifact_sha256"] == cl["artifact_sha256"], "outputs/eval/RUN_RECORD.json")
    cc = json.loads(Path("configs/classical_final_dev.json").read_text())
    chk("classical frozen config equals configs/classical_final_dev.json", cl["config"] == cc["config"] and cl["config_sha"] == cc["config_sha"] == "6696e8cab3ba", "configs/frozen/classical_v1.json")
    # counts
    s1d = pd.read_csv("experiments/s1_same_protocol_per_image.csv"); cld = pd.read_csv("experiments/classical_final_per_image.csv")
    ex = lambda d, c: int((d[c] == d.expected).sum())
    chk("dev S1 exact count 73/88 = 0.830", ex(s1d, "n_pred") == 73 and len(s1d) == 88, "experiments/s1_same_protocol_per_image.csv", f"{ex(s1d,'n_pred')}/{len(s1d)}")
    chk("dev classical exact count 42/88 = 0.477", ex(cld, "n_pred") == 42 and len(cld) == 88, "experiments/classical_final_per_image.csv", f"{ex(cld,'n_pred')}/{len(cld)}")
    e = pd.read_csv("experiments/final_eval_with_counts.csv")
    chk("eval S1 exact 19/22, classical 12/22", int((e.s1_n == e.expected_egg_count).sum()) == 19 and int((e.classical_n == e.expected_egg_count).sum()) == 12, "experiments/final_eval_with_counts.csv")
    chk("eval predicted/expected totals S1 22, classical 26, expected 25", (e.s1_n.sum(), e.classical_n.sum(), e.expected_egg_count.sum()) == (22, 26, 25), "experiments/final_eval_with_counts.csv")
    chk("dev S1 over 1/88, under 14/88", int((s1d.n_pred > s1d.expected).sum()) == 1 and int((s1d.n_pred < s1d.expected).sum()) == 14, "experiments/s1_same_protocol_per_image.csv")
    inst = (int(cld.rev_inst_tp.sum()), int(cld.rev_inst_fp.sum()), int(cld.rev_inst_fn.sum()))
    p, r = inst[0] / (inst[0] + inst[1]), inst[0] / (inst[0] + inst[2])
    chk("dev classical inst P/R/F1 vs R_rev = 0.442/0.393/0.416", (round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3)) == (0.442, 0.393, 0.416), "experiments/classical_final_per_image.csv", f"{inst}")
    ne = cld[~cld.rev_ref_empty]
    chk("dev classical macro IoU/Dice vs R_rev = 0.363/0.441", (round(ne.rev_iou.fillna(0).mean(), 3), round(ne.rev_dice.fillna(0).mean(), 3)) == (0.363, 0.441), "experiments/classical_final_per_image.csv", f"n={len(ne)}")
    ne2 = e[~e.agree_ref_empty]
    chk("eval agreement macro IoU/Dice vs S1 = 0.315/0.439 over 20", (len(ne2), round(ne2.agree_iou_vs_S1.mean(), 3), round(ne2.agree_dice_vs_S1.mean(), 3)) == (20, 0.315, 0.439), "experiments/final_eval_with_counts.csv")
    chk("eval instance TP/FP/FN vs S1 = 6/20/16", (int(e.inst_tp_vs_S1.sum()), int(e.inst_fp_vs_S1.sum()), int(e.inst_fn_vs_S1.sum())) == (6, 20, 16), "experiments/final_eval_with_counts.csv")
    run = json.loads(Path("outputs/sam_candidates/run.json").read_text()); aud = json.loads(Path("outputs/sam_candidates/audit.json").read_text())
    chk("14,400 candidates and 1803.6 s / 110 = 16.4 s per image", aud["stats"]["n_candidates_manifest"] == 14400 and round(run["total_runtime_s"] / 110, 1) == 16.4, "outputs/sam_candidates/audit.json, run.json")
    chk("S1 frozen SAM checkpoint sha equals run.json and file on disk", s1["sam"]["checkpoint_sha256"] == run["checkpoint_sha256"], "configs/frozen/S1_v1.json")
    import hashlib
    chk("checkpoint on disk matches recorded sha256", hashlib.sha256(Path("models/sam_vit_b_01ec64.pth").read_bytes()).hexdigest() == run["checkpoint_sha256"], "models/sam_vit_b_01ec64.pth")
    chk("frozen environment torch/python vs venv", s1["environment"]["torch"] == run["torch"], "configs/frozen/S1_v1.json", f"{s1['environment']['torch']} python {s1['environment']['python']}")
    # report text
    t = "".join(p.get_text() for p in fitz.open("report/REPORT.pdf"))
    for h in (s1["artifact_sha256"][:16], cl["artifact_sha256"][:16], cl["config_sha"]):
        chk(f"report contains hash {h}", h in t, "report/REPORT.pdf")
    for s in ("0.830 (73/88)", "0.864 (19/22)", "0.477 (42/88)", "0.545 (12/22)", "0.442/0.393/0.416", "22 / 25", "26 / 25", "15 are empty/flagged: 13 dev, 2 eval", "213 pass"):
        chk(f"report contains '{s}'", s.replace(" ", "") in t.replace(" ", "").replace("\n", ""), "report/REPORT.pdf")
    # referenced paths
    docs = ["README.md", "experiments/video_tracking_results.md", "experiments/video_tracking_protocol.md", "experiments/final_evaluation_analysis.md", "experiments/final_evaluation_results.md", "experiments/final_evaluation_protocol.md", "experiments/classical_failure_analysis.md"]
    txt = {d: Path(d).read_text(encoding="utf-8") for d in docs}; txt["report/REPORT.pdf"] = t
    ignore = re.compile(r"data_raw|scratch/|<|\*|Exx|egg_tracking_T1\.mp4|sam_vit_b|^\.venv|^requirements|\.\.\.")
    missing = {}
    for d, s in txt.items():
        for m in set(re.findall(r"[A-Za-z0-9_./-]+\.(?:py|md|csv|json|jpg|png|txt|pdf)\b", s)):
            if ignore.search(m) or m.startswith("http") or "/" not in m and not Path(m).exists() and m.endswith(("_per_image.csv",)) is False and False:
                continue
            if "/" in m and not Path(m).exists():
                missing.setdefault(d, []).append(m)
    chk("every path with a directory referenced in README/reports exists", not missing, "README.md, experiments/*.md, report/REPORT.pdf", json.dumps(missing))
    Path("experiments/consistency_checks.json").write_text(json.dumps(R, indent=1))
    for r in R:
        print(r["result"], "|", r["check"], "|", r["detail"])


if __name__ == "__main__":
    main()
