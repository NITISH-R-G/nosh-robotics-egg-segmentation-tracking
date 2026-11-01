import json

import cv2
import numpy as np
import pandas as pd
import pytest

from src.experiments.review_pool import mask_sha256
from src.segmentation.build_reviewed import ADD, _dir_digest, build_reviewed
from src.segmentation.cache_candidates import candidate_record, pack_masks
from src.segmentation.masks import load_mask_png, save_mask_png

H, W = 80, 120


def _ell(cx, cy, a, b):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, ((cx, cy), (2 * a, 2 * b), 0), 1, -1)
    return m.astype(bool)


@pytest.fixture
def world(tmp_path):
    cache, raw = tmp_path / "cache", tmp_path / "raw"
    (cache / "masks").mkdir(parents=True)
    egg_a, egg_b = _ell(30, 40, 15, 10), _ell(90, 40, 15, 10)
    lines = []
    for img, ms in {"img_1": [egg_a, egg_b], "img_2": [egg_a], "img_9": [egg_a]}.items():
        packed, shape = pack_masks(ms)
        np.savez_compressed(cache / "masks" / f"{img}.npz", masks=packed, shape=np.array(shape))
        for k, m in enumerate(ms):
            ys, xs = np.nonzero(m)
            c = {"bbox_xywh": [xs.min(), ys.min(), 1, 1], "predicted_iou": 0.9, "stability_score": 0.9, "point_coords": [1.0, 1.0], "crop_box": [0, 0, W, H]}
            lines.append(json.dumps(candidate_record(img, k, f"masks/{img}.npz", m, c)))
    (cache / "candidates.jsonl").write_text("\n".join(lines) + "\n")
    save_mask_png(raw / "img_1.png", egg_a)          # S1 found egg A, missed egg B
    save_mask_png(raw / "img_2.png", np.zeros((H, W), bool))  # S1 selected nothing
    split = tmp_path / "split.csv"
    pd.DataFrame({"image_id": ["img_1", "img_2", "img_9"], "split": ["dev", "dev", "eval"]}).to_csv(split, index=False)
    man = pd.DataFrame([
        {"image_id": "img_1", "review_id": "D1-c1", "candidate_id": "img_1#1", "mask_sha256": mask_sha256(egg_b), "action": ADD},
        {"image_id": "img_2", "review_id": "D2-c0", "candidate_id": "img_2#0", "mask_sha256": mask_sha256(egg_a), "action": "exclude"}])
    return tmp_path, cache, raw, split, man, egg_a, egg_b


def test_reviewed_adds_only_marked_candidates_and_never_touches_raw(world):
    tmp, cache, raw, split, man, a, b = world
    before = _dir_digest(raw)
    out = build_reviewed(man, cache, raw, tmp / "rev", split)
    assert _dir_digest(raw) == before                                    # raw masks bit-identical
    assert (load_mask_png(tmp / "rev" / "masks" / "img_1.png").astype(bool) == (a | b)).all()
    assert not load_mask_png(tmp / "rev" / "masks" / "img_2.png").any()  # excluded row is not added
    r = out.set_index("image_id")
    assert r.loc["img_1", "changed"] and not r.loc["img_2", "changed"] and r.loc["img_1", "added_candidate_ids"] == ["img_1#1"]
    assert r.loc["img_1", "raw_mask_sha256"] == mask_sha256(a) and r.loc["img_1", "reviewed_mask_sha256"] == mask_sha256(a | b)
    assert list(out.image_id) == ["img_1", "img_2"]  # dev only; the eval image is never produced


def test_refuses_eval_images_wrong_hash_overwrite_and_bad_output_dir(world):
    tmp, cache, raw, split, man, a, b = world
    with pytest.raises(ValueError, match="non-development"):
        build_reviewed(man.assign(image_id=["img_9", "img_2"], candidate_id=["img_9#0", "img_2#0"]), cache, raw, tmp / "o1", split)
    bad = man.copy()
    bad.loc[0, "mask_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash differs"):
        build_reviewed(bad, cache, raw, tmp / "o2", split)
    with pytest.raises(ValueError, match="separate"):
        build_reviewed(man, cache, raw, raw / "reviewed", split)
    with pytest.raises(ValueError, match="separate"):
        build_reviewed(man, cache, raw, cache / "reviewed", split)
    build_reviewed(man, cache, raw, tmp / "o3", split)
    with pytest.raises(FileExistsError):
        build_reviewed(man, cache, raw, tmp / "o3", split)
    with pytest.raises(ValueError, match="duplicate"):
        build_reviewed(pd.concat([man, man.iloc[[0]]], ignore_index=True), cache, raw, tmp / "o4", split)
