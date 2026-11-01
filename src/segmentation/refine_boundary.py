"""Box-prompted SAM boundary refinement for an explicit, closed list of development proposals.

Protocol: experiments/boundary_refinement_protocol.md (written before this code ran).
Separate path: the candidate cache and all S1 outputs are read-only; refined masks are written under their own directory
with NEW ids (R1:/R2: prefix); an original candidate id is never reused for a changed mask; the experiment runs once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.experiments.diagnose import assert_dev
from src.experiments.review_pool import mask_sha256, verify_identity
from src.segmentation.cache_candidates import read_candidates, unpack_masks
from src.segmentation.masks import save_mask_png

VARIANTS = {"V1": 0.0, "V2": 0.08}  # box padding fraction, fixed by the protocol
PROTOCOL = Path("experiments/boundary_refinement_protocol.md")


def prompt_box(bbox_xywh, pad_frac: float, h: int, w: int) -> np.ndarray:
    """Cached SAM bbox (x, y, w, h) -> xyxy box, padded by pad_frac of its width/height on each side, clipped to the image."""
    x, y, bw, bh = (float(v) for v in bbox_xywh)
    px, py = pad_frac * bw, pad_frac * bh
    return np.array([max(x - px, 0.0), max(y - py, 0.0), min(x + bw + px, w - 1.0), min(y + bh + py, h - 1.0)], dtype=np.float32)


def refined_id(variant: str, image_id: str, index: int) -> str:
    """New id for a refined mask; never equal to the original candidate id `<image_id>#<index>`."""
    return f"R{variant[1:]}:{image_id}#{index}"


def _file_stem(variant: str, image_id: str, index: int) -> str:
    return f"{image_id}__{index}"  # ':' is not allowed in Windows file names


def code_version() -> str:
    h = hashlib.sha256(Path(__file__).read_bytes())
    if PROTOCOL.exists():
        h.update(PROTOCOL.read_bytes())
    return h.hexdigest()[:12]


def refine_cases(cases: pd.DataFrame, cache: Path, data: Path, split_csv: Path, out_dir: Path, predictor,
                 variants: dict[str, float] = VARIANTS, device: str = "cpu", sync=lambda: None, peak_mem=lambda: None,
                 reset_peak=lambda: None, extra_meta: dict | None = None) -> pd.DataFrame:
    """predictor: object with set_image(rgb) and predict(box=..., multimask_output=False) -> (masks, scores, logits)."""
    cache, out_dir = Path(cache), Path(out_dir)
    if cache.resolve() in out_dir.resolve().parents or out_dir.resolve() == cache.resolve():
        raise ValueError("refined masks must be written outside the candidate cache")
    done = [out_dir / v / "manifest.jsonl" for v in variants if (out_dir / v / "manifest.jsonl").exists()]
    if done:
        raise FileExistsError(f"{done[0]} exists: the protocol allows each variant to run exactly once")
    image_ids = sorted(cases.image_id.unique())
    assert_dev(image_ids, Path(split_csv))
    sp = pd.read_csv(split_csv).set_index("image_id")
    cands = read_candidates(cache, set(image_ids))
    rows = []
    for img_id in image_ids:
        masks = unpack_masks(cache / "masks" / f"{img_id}.npz")
        img = cv2.imdecode(np.fromfile(str(data / sp.loc[img_id, "path"]), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if masks.shape[1:] != (h, w):
            raise ValueError(f"{img_id}: cached masks {masks.shape[1:]} != image {(h, w)}")
        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for case in cases[cases.image_id == img_id].itertuples():
            idx = int(case.candidate_index)
            verify_identity(cands, masks, img_id, idx)  # the cached mask really is this candidate
            if mask_sha256(masks[idx]) != case.mask_sha256:
                raise ValueError(f"{case.candidate_id}: cached mask hash differs from the case list; refusing to refine")
            rec = cands[(cands.image_id == img_id) & (cands["index"] == idx)].iloc[0]
            for vname, pad in variants.items():
                box = prompt_box(rec["bbox_xywh"], pad, h, w)
                reset_peak()
                sync()
                t0 = time.perf_counter()
                m, scores, _ = predictor.predict(box=box, multimask_output=False)
                sync()
                dt = time.perf_counter() - t0
                refined = np.asarray(m[0], bool)
                new_id = refined_id(vname, img_id, idx)
                assert new_id != case.candidate_id
                inter = float((refined & masks[idx]).sum())
                union = float((refined | masks[idx]).sum())
                save_mask_png(out_dir / vname / "masks" / f"{_file_stem(vname, img_id, idx)}.png", refined)
                rows.append({"variant": vname, "refined_id": new_id, "image_id": img_id, "review_id": case.review_id,
                             "original_candidate_id": case.candidate_id, "original_mask_sha256": case.mask_sha256,
                             "refined_mask_sha256": mask_sha256(refined), "prompt_box_xyxy": [round(float(v), 2) for v in box],
                             "box_padding_fraction": pad, "multimask_output": False, "sam_predicted_iou": round(float(scores[0]), 4),
                             "area_original": int(masks[idx].sum()), "area_refined": int(refined.sum()),
                             "area_ratio": round(float(refined.sum()) / max(float(masks[idx].sum()), 1.0), 4),
                             "agreement_iou_with_original": round(inter / union, 4) if union else 0.0,
                             "runtime_s": round(dt, 4), "peak_gpu_mb": peak_mem(), "device": device,
                             "code_version": code_version(), **(extra_meta or {})})
    man = pd.DataFrame(rows)
    for vname in variants:
        d = out_dir / vname
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "manifest.jsonl", "w") as f:
            for r in man[man.variant == vname].to_dict("records"):
                f.write(json.dumps(r) + "\n")
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="experiments/boundary_refinement_cases.csv")
    ap.add_argument("--cache", default="outputs/sam_candidates")
    ap.add_argument("--data", default="data_raw")
    ap.add_argument("--split", default="data/split_manifest.csv")
    ap.add_argument("--checkpoint", default="models/sam_vit_b_01ec64.pth")
    ap.add_argument("--out", default="outputs/boundary_refinement")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    from src.segmentation.run_sam import sha256_of

    torch.manual_seed(0)
    sam = sam_model_registry["vit_b"](checkpoint=a.checkpoint).to(a.device).eval()
    pred = SamPredictor(sam)
    cuda = a.device == "cuda" and torch.cuda.is_available()
    meta = {"checkpoint_sha256": sha256_of(Path(a.checkpoint)), "torch": torch.__version__}
    cases = pd.read_csv(a.cases)
    t0 = time.perf_counter()
    man = refine_cases(cases, Path(a.cache), Path(a.data), Path(a.split), Path(a.out), pred, device=a.device,
                       sync=(torch.cuda.synchronize if cuda else (lambda: None)),
                       reset_peak=(torch.cuda.reset_peak_memory_stats if cuda else (lambda: None)),
                       peak_mem=(lambda: round(torch.cuda.max_memory_allocated() / 2**20, 1)) if cuda else (lambda: None), extra_meta=meta)
    print(len(man), "refined masks in", round(time.perf_counter() - t0, 1), "s ->", a.out)
    print(man.groupby("variant").agg(n=("refined_id", "size"), runtime_median_s=("runtime_s", "median"), peak_gpu_mb=("peak_gpu_mb", "max")))


if __name__ == "__main__":
    main()
