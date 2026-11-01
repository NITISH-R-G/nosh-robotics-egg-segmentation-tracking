import json

import cv2
import numpy as np
import pandas as pd
import pytest

from src.experiments.review_pool import mask_sha256
from src.segmentation.cache_candidates import candidate_record, pack_masks
from src.segmentation.refine_boundary import VARIANTS, prompt_box, refine_cases, refined_id

H, W = 100, 150


def _ell(cx, cy, a, b):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, ((cx, cy), (2 * a, 2 * b), 0), 1, -1)
    return m.astype(bool)


class StubPredictor:
    """Records what it was asked and returns a deliberately different (shifted) mask."""

    def __init__(self):
        self.calls, self.image = [], None

    def set_image(self, rgb):
        assert rgb.shape == (H, W, 3)
        self.image = rgb

    def predict(self, box, multimask_output):
        self.calls.append((np.asarray(box).copy(), multimask_output))
        m = np.zeros((H, W), bool)
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        m[y0:y1 + 1, x0:x1 + 1] = True
        return m[None], np.array([0.9]), None


@pytest.fixture
def world(tmp_path):
    cache = tmp_path / "cache"
    (cache / "masks").mkdir(parents=True)
    ms = {"img_1": [_ell(40, 50, 20, 14), _ell(110, 50, 15, 10)], "img_5": [_ell(75, 50, 20, 14)]}
    lines = []
    for img, masks in ms.items():
        cv2.imwrite(str(tmp_path / f"{img}.jpg"), np.zeros((H, W, 3), np.uint8))
        packed, shape = pack_masks(masks)
        np.savez_compressed(cache / "masks" / f"{img}.npz", masks=packed, shape=np.array(shape))
        for k, m in enumerate(masks):
            ys, xs = np.nonzero(m)
            c = {"bbox_xywh": [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1], "predicted_iou": 0.9,
                 "stability_score": 0.9, "point_coords": [1.0, 1.0], "crop_box": [0, 0, W, H]}
            lines.append(json.dumps(candidate_record(img, k, f"masks/{img}.npz", m, c)))
    (cache / "candidates.jsonl").write_text("\n".join(lines) + "\n")
    split = tmp_path / "split.csv"
    pd.DataFrame({"image_id": ["img_1", "img_5"], "split": ["dev", "eval"], "path": ["img_1.jpg", "img_5.jpg"]}).to_csv(split, index=False)
    cases = pd.DataFrame([{"review_id": f"D1-c{k}", "image_id": "img_1", "candidate_id": f"img_1#{k}", "candidate_index": k,
                           "mask_sha256": mask_sha256(ms["img_1"][k])} for k in (0, 1)])
    return tmp_path, cache, split, cases, ms


def test_prompt_box_from_cached_bbox_padding_and_clipping():
    assert prompt_box([10, 20, 30, 40], 0.0, 100, 150).tolist() == [10, 20, 40, 60]
    assert prompt_box([10, 20, 30, 40], 0.1, 100, 150).tolist() == [7, 16, 43, 64]
    clipped = prompt_box([0, 0, 149, 99], 0.08, 100, 150)
    assert clipped.min() >= 0 and clipped[2] <= 149 and clipped[3] <= 99
    assert VARIANTS == {"V1": 0.0, "V2": 0.08}  # protocol values


def test_refined_ids_never_equal_original_ids():
    assert refined_id("V1", "img_1", 4) == "R1:img_1#4" and refined_id("V2", "img_1", 4) == "R2:img_1#4"
    assert refined_id("V1", "img_1", 4) != "img_1#4"


def test_refinement_writes_separately_records_provenance_and_keeps_cache_intact(world):
    tmp, cache, split, cases, ms = world
    before = {p.name: p.read_bytes() for p in list(cache.glob("**/*")) if p.is_file()}
    pred = StubPredictor()
    out = tmp / "out"
    man = refine_cases(cases, cache, tmp, split, out, pred, extra_meta={"checkpoint_sha256": "x"})
    assert len(man) == 4 and set(man.variant) == {"V1", "V2"} and all(c[1] is False for c in pred.calls)
    assert {p.name: p.read_bytes() for p in cache.glob("**/*") if p.is_file()} == before  # cache bit-identical
    r = man[(man.variant == "V1") & (man.review_id == "D1-c0")].iloc[0]
    assert r.refined_id == "R1:img_1#0" and r.original_candidate_id == "img_1#0" and r.original_mask_sha256 == mask_sha256(ms["img_1"][0])
    assert r.refined_mask_sha256 != r.original_mask_sha256 and r.box_padding_fraction == 0.0 and len(r.code_version) == 12
    assert (out / "V1" / "masks" / "img_1__0.png").exists() and (out / "V2" / "manifest.jsonl").exists()
    # V2 box is larger than V1 box for the same case
    b1 = np.array(r.prompt_box_xyxy)
    b2 = np.array(man[(man.variant == "V2") & (man.review_id == "D1-c0")].iloc[0].prompt_box_xyxy)
    assert b2[0] < b1[0] and b2[2] > b1[2]


def test_refuses_eval_images_cache_output_rerun_and_hash_mismatch(world):
    tmp, cache, split, cases, ms = world
    evalcase = cases.assign(image_id="img_5", candidate_id="img_5#0", candidate_index=0)
    with pytest.raises(ValueError, match="non-development"):
        refine_cases(evalcase, cache, tmp, split, tmp / "o1", StubPredictor())
    with pytest.raises(ValueError, match="outside the candidate cache"):
        refine_cases(cases, cache, tmp, split, cache / "refined", StubPredictor())
    bad = cases.copy()
    bad.loc[0, "mask_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash differs"):
        refine_cases(bad, cache, tmp, split, tmp / "o2", StubPredictor())
    refine_cases(cases, cache, tmp, split, tmp / "o3", StubPredictor())
    first = (tmp / "o3" / "V1" / "manifest.jsonl").read_bytes()
    with pytest.raises(FileExistsError, match="exactly once"):  # a genuine second run into the same directory is refused
        refine_cases(cases, cache, tmp, split, tmp / "o3", StubPredictor())
    assert (tmp / "o3" / "V1" / "manifest.jsonl").read_bytes() == first  # and the first results are not overwritten
