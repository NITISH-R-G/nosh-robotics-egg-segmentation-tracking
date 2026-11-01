import json

import cv2
import numpy as np
import pandas as pd
import pytest

from src.evaluation.references import ADD, load_references
from src.segmentation.cache_candidates import candidate_record, pack_masks
from src.segmentation.masks import save_mask_png

H, W = 60, 90


def _ell(cx, cy, a, b):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, ((cx, cy), (2 * a, 2 * b), 0), 1, -1)
    return m.astype(bool)


@pytest.fixture
def world(tmp_path):
    cache, raw, rev = tmp_path / "cache", tmp_path / "raw", tmp_path / "rev"
    (cache / "masks").mkdir(parents=True)
    a, b = _ell(20, 30, 12, 8), _ell(65, 30, 12, 8)
    lines = []
    for img, ms in {"i1": [a, b], "i2": [a], "i3": [b]}.items():
        packed, shape = pack_masks(ms)
        np.savez_compressed(cache / "masks" / f"{img}.npz", masks=packed, shape=np.array(shape))
        for k, m in enumerate(ms):
            ys, xs = np.nonzero(m)
            lines.append(json.dumps(candidate_record(img, k, f"masks/{img}.npz", m, {
                "bbox_xywh": [xs.min(), ys.min(), 1, 1], "predicted_iou": 0.9, "stability_score": 0.9, "point_coords": [1.0, 1.0], "crop_box": [0, 0, W, H]})))
    (cache / "candidates.jsonl").write_text("\n".join(lines) + "\n")
    save_mask_png(raw / "i1.png", a)                 # S1 selected egg a only
    save_mask_png(raw / "i2.png", np.zeros((H, W), bool))  # S1 selected nothing
    save_mask_png(rev / "i1.png", a | b)             # manual addition of b
    save_mask_png(rev / "i2.png", a)                 # manual addition of a
    run = tmp_path / "run.csv"
    pd.DataFrame({"image_id": ["i1", "i2"], "selected_idx": ["[0]", "[]"]}).to_csv(run, index=False)
    man = tmp_path / "man.csv"
    pd.DataFrame({"image_id": ["i1", "i2", "i2"], "candidate_id": ["i1#1", "i2#0", "i2#0"],
                  "action": [ADD, ADD, "exclude"]}).drop(2).to_csv(man, index=False)
    split = tmp_path / "split.csv"
    pd.DataFrame({"image_id": ["i1", "i2", "i3"], "split": ["dev", "dev", "eval"]}).to_csv(split, index=False)
    return tmp_path, cache, raw, rev, run, man, split, a, b


def test_instances_reconstructed_and_references_kept_separate(world):
    tmp, cache, raw, rev, run, man, split, a, b = world
    r = load_references(["i1", "i2"], split, raw, rev, run, man, cache)
    assert len(r["i1"]["raw_instances"]) == 1 and len(r["i1"]["rev_instances"]) == 2
    assert r["i1"]["raw_candidate_ids"] == ["i1#0"] and r["i1"]["rev_candidate_ids"] == ["i1#0", "i1#1"]
    assert not r["i2"]["raw_union"].any() and r["i2"]["rev_union"].any()  # empty in raw, filled in reviewed
    assert (r["i1"]["raw_union"] == a).all() and (r["i1"]["rev_union"] == (a | b)).all()


def test_mismatch_between_stored_mask_and_instances_is_rejected(world):
    tmp, cache, raw, rev, run, man, split, a, b = world
    save_mask_png(rev / "i1.png", a)  # stored reviewed mask no longer equals raw + addition
    with pytest.raises(ValueError, match="do not reproduce"):
        load_references(["i1"], split, raw, rev, run, man, cache)


def test_eval_images_rejected(world):
    tmp, cache, raw, rev, run, man, split, a, b = world
    with pytest.raises(ValueError, match="non-development"):
        load_references(["i1", "i3"], split, raw, rev, run, man, cache)
