"""Freeze S1 and the chosen classical configuration as two separate, hashed artifacts (configs/frozen/).

Run once, before the evaluation split is used. After freezing, any change needs a NEW version file (S1_v2.json ...) and a logged reason;
final_eval.py refuses to run if a recorded source/evidence hash differs.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

from src.segmentation.frozen import canonical_sha256
from src.segmentation.selector import SelectorConfig

FROZEN = Path("configs/frozen")


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def seal(doc: dict) -> dict:
    doc = {k: v for k, v in doc.items() if k != "artifact_sha256"}
    doc["artifact_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return doc


def env() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "requirements_lock_sha256": sha256("requirements.lock.txt")}


def main() -> None:
    if (FROZEN / "S1_v1.json").exists() or (FROZEN / "classical_v1.json").exists():
        raise RuntimeError("v1 already frozen; create a new version and log the reason instead of overwriting")
    FROZEN.mkdir(parents=True, exist_ok=True)
    run = json.loads(Path("outputs/sam_candidates/run.json").read_text())
    audit = json.loads(Path("outputs/sam_candidates/audit.json").read_text())
    protocol = "experiments/final_evaluation_protocol.md"

    # ---- S1 -------------------------------------------------------------------------------------------------------
    sel = SelectorConfig("S1_conservative_plus_review_flag", min_area_frac=0.01, max_area_frac=0.60, min_solidity=0.95, max_aspect=2.0,
                         min_ellipse_iou=0.93, min_pred_iou=0.90, min_stability=0.90, fallback="review")
    sel.validate()
    d = sel.as_dict()
    s1 = {"name": "S1_v1", "kind": "pretrained SAM candidates + feature selector", "selector": d, "selector_sha256": canonical_sha256(d),
          "pipeline": ["SAM ViT-B SamAutomaticMaskGenerator on the original RGB image (no resize/preprocessing beyond SAM's own)",
                       "candidate features (predicted_iou, stability, solidity, aspect, ellipse_iou, area) from the cached candidates",
                       "gates + NMS in src/segmentation/selector.py; fallback='review' only FLAGS images with no selection (selections identical to S1)"],
          "sam": {"checkpoint": run.get("checkpoint"), "checkpoint_sha256": run.get("checkpoint_sha256"), "model": run.get("model"),
                  "generator": audit["stats"]["generator"], "device_at_cache_time": run.get("device"), "torch_at_cache_time": run.get("torch")},
          "candidate_cache_hashes": audit["stats"]["hashes"], "selected_on": "88 development images only (selector_log.csv, fallback_evaluation.md); thresholds are fixed hypotheses",
          "label_caveat": "evidence labels are AI-assisted and unverified by a human; reference masks are SAM-derived pseudo-ground-truth",
          "source_files_sha256": {f: sha256(f) for f in ("src/segmentation/selector.py", "src/segmentation/sam_segmenter.py", "src/segmentation/cache_candidates.py",
                                                         "src/segmentation/make_reference.py", "src/segmentation/frozen.py", "src/segmentation/masks.py",
                                                         "src/experiments/final_eval.py", "src/evaluation/metrics.py")},
          "evidence_sha256": {"data/split_manifest.csv": sha256("data/split_manifest.csv"), "outputs/sam_candidates/audit.json": sha256("outputs/sam_candidates/audit.json"),
                              "outputs/sam_candidates/candidates.jsonl": sha256("outputs/sam_candidates/candidates.jsonl"), protocol: sha256(protocol)},
          "environment": env(), "supersedes": "configs/withdrawn/selector_frozen_WITHDRAWN_20261029.json (withdrawn freeze; never used to make references)"}
    (FROZEN / "S1_v1.json").write_text(json.dumps(seal(s1), indent=1, sort_keys=True))

    # ---- classical ------------------------------------------------------------------------------------------------
    c = json.loads(Path("configs/classical_final_dev.json").read_text())
    cl = {"name": "classical_v1", "kind": "classical image processing (no learned components)", "config": c["config"], "config_sha": c["config_sha"],
          "pipeline": ["BGR uint8 image, no resizing", "foreground map: Lab lightness minus chroma ('W')", "global Otsu threshold",
                       "5x5 open, 5x5 close, hole fill, 8-connected components", "filters: area 1%-60%, aspect <= 2.2, border-touching components >= 5% of the image"],
          "selected_on": "88 development images only by the pre-registered greedy rule (experiments/classical_comparison_protocol*.md, classical_log.csv)",
          "recommendation_recorded": "documented simple baseline, NOT presented as competitive (experiments/classical_failure_analysis.md)",
          "label_caveat": "development scores are against SAM-derived references with AI-assisted corrections",
          "source_files_sha256": {f: sha256(f) for f in ("src/segmentation/classical.py", "src/experiments/classical_experiments.py", "src/evaluation/metrics.py",
                                                         "src/segmentation/masks.py", "src/experiments/final_eval.py")},
          "evidence_sha256": {"data/split_manifest.csv": sha256("data/split_manifest.csv"), "experiments/classical_log.csv": sha256("experiments/classical_log.csv"),
                              "configs/classical_final_dev.json": sha256("configs/classical_final_dev.json"), protocol: sha256(protocol)},
          "environment": env()}
    (FROZEN / "classical_v1.json").write_text(json.dumps(seal(cl), indent=1, sort_keys=True))
    print("S1_v1", json.loads((FROZEN / "S1_v1.json").read_text())["artifact_sha256"][:16])
    print("classical_v1", json.loads((FROZEN / "classical_v1.json").read_text())["artifact_sha256"][:16])


if __name__ == "__main__":
    main()
