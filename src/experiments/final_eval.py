"""Run each FROZEN method exactly once on the evaluation split (22 images) and write separate outputs.

Guards: both frozen files must verify their hashes, the split manifest and the cache audit must match the freeze
record, and the run refuses to execute a second time (outputs/eval/RUN_RECORD.json). No threshold is read from
or changed by anything computed here. Evaluation outputs live under outputs/eval/ and never touch development outputs.

S1 on eval = the frozen selector applied to the already-cached raw SAM candidates (no new SAM pass is needed).
Eval has no reference except S1's own raw masks, so the S1 vs classical comparison below is AGREEMENT with an S1-derived
reference, not accuracy; S1 scored against itself is trivially 1.0 and is not reported as a result.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.evaluation.metrics import connected_instances, match_instances, pixel_metrics
from src.segmentation.classical import ClassicalConfig, segment_classical
from src.segmentation.make_reference import make_reference
from src.segmentation.masks import save_mask_png

OUT = Path("outputs/eval")
FROZEN = Path("configs/frozen")


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_freeze() -> tuple[dict, dict]:
    s1 = json.loads((FROZEN / "S1_v1.json").read_text())
    cl = json.loads((FROZEN / "classical_v1.json").read_text())
    for doc in (s1, cl):
        body = {k: v for k, v in doc.items() if k != "artifact_sha256"}
        if hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != doc["artifact_sha256"]:
            raise RuntimeError(f"frozen artifact {doc['name']} was modified after freezing")
        for path, h in doc["source_files_sha256"].items():
            if sha256(path) != h:
                raise RuntimeError(f"source file changed since freeze: {path}")
    for path, h in {**s1["evidence_sha256"], **cl["evidence_sha256"]}.items():
        if sha256(path) != h:
            raise RuntimeError(f"evidence/input file changed since freeze: {path}")
    return s1, cl


def main() -> None:
    record = OUT / "RUN_RECORD.json"
    if record.exists():
        raise RuntimeError("final evaluation already ran once; a second run needs a new frozen version and a logged reason")
    s1_doc, cl_doc = verify_freeze()
    split = pd.read_csv("data/split_manifest.csv")
    ev = split[split.split == "eval"].sort_values("image_id")
    assert len(ev) == 22, len(ev)
    OUT.mkdir(parents=True, exist_ok=True)
    ev.to_csv(OUT / "eval_split.csv", index=False)

    t0 = time.perf_counter()
    man = make_reference(Path("outputs/sam_candidates"), OUT / "eval_split.csv", FROZEN / "S1_v1.json", OUT / "S1", Path("outputs/sam_candidates/audit.json"))
    s1_total_s = time.perf_counter() - t0

    ccfg = ClassicalConfig(**cl_doc["config"])
    assert ccfg.sha() == cl_doc["config_sha"]
    rows = []
    for r in ev.itertuples():
        img = cv2.imdecode(np.fromfile(str(Path("data_raw") / r.path), np.uint8), cv2.IMREAD_COLOR)
        t1 = time.perf_counter()
        union, inst, info = segment_classical(img, ccfg)
        dt = time.perf_counter() - t1
        save_mask_png(OUT / "classical" / "masks" / f"{r.image_id}.png", union.astype(bool))
        s1_mask = cv2.imdecode(np.fromfile(str(OUT / "S1" / "masks" / f"{r.image_id}.png"), np.uint8), cv2.IMREAD_GRAYSCALE) > 0
        s1_inst = connected_instances(s1_mask)
        s1_lab = cv2.imdecode(np.fromfile(str(OUT / "S1" / "instances" / f"{r.image_id}.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
        s1_inst = [s1_lab == k for k in range(1, int(s1_lab.max()) + 1)]
        pm = pixel_metrics(s1_mask, union)
        mt = match_instances(inst, s1_inst, 0.5)
        mrow = man[man.image_id == r.image_id].iloc[0]
        rows.append({"image_id": r.image_id, "s1_n": int(mrow.detection_count), "s1_status": mrow.status, "s1_needs_review": bool(mrow.needs_review),
                     "classical_n": len(inst), "classical_runtime_s": dt, "agree_ref_empty": pm["ref_empty"], "agree_pred_empty": pm["pred_empty"],
                     "agree_iou_vs_S1": pm["iou"], "agree_dice_vs_S1": pm["dice"], "inst_tp_vs_S1": mt["tp"], "inst_fp_vs_S1": mt["fp"], "inst_fn_vs_S1": mt["fn"]})
    res = pd.DataFrame(rows)
    res.to_csv("experiments/final_eval_per_image.csv", index=False)
    record.write_text(json.dumps({"ran_at": time.strftime("%Y%m%d-%H%M%S"), "n_images": len(ev), "s1_selector_total_s": s1_total_s,
                                  "s1_artifact_sha256": s1_doc["artifact_sha256"], "classical_artifact_sha256": cl_doc["artifact_sha256"],
                                  "per_image_csv_sha256": sha256("experiments/final_eval_per_image.csv")}, indent=1))
    print(res.describe(include="all").T[["count", "mean"]])
    print("S1 total", round(s1_total_s, 2), "s; classical median", round(float(res.classical_runtime_s.median()), 3), "s")


if __name__ == "__main__":
    main()
