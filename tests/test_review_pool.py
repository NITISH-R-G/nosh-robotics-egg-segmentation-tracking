import json

import cv2
import numpy as np
import pandas as pd
import pytest

from src.experiments.review_pool import build_items, mask_sha256, render_panel
from src.segmentation.cache_candidates import candidate_record, pack_masks

H = W = 40


def _m(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


@pytest.fixture
def setup(tmp_path):
    cache = tmp_path / "cache"
    (cache / "masks").mkdir(parents=True)
    # candidates 0 and 2 are identical masks; candidate 1 is merely similar (one pixel different)
    masks = [_m(5, 15, 5, 15), _m(5, 15, 5, 16), _m(5, 15, 5, 15)]
    packed, shape = pack_masks(masks)
    np.savez_compressed(cache / "masks" / "not_damaged_7.npz", masks=packed, shape=np.array(shape))
    rows = []
    for k, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        c = {"bbox_xywh": [xs.min(), ys.min(), 10, 10], "predicted_iou": 0.9, "stability_score": 0.9,
             "point_coords": [1.0, 1.0], "crop_box": [0, 0, W, H]}
        rows.append(candidate_record("not_damaged_7", k, "masks/not_damaged_7.npz", m, c))
    (cache / "candidates.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    split = tmp_path / "split.csv"
    pd.DataFrame({"image_id": ["not_damaged_7", "not_damaged_9"], "split": ["dev", "eval"],
                  "path": ["a.jpg", "b.jpg"]}).to_csv(split, index=False)

    def run(name, idx, img="not_damaged_7"):
        p = tmp_path / f"selector_{name}_per_image.csv"
        pd.DataFrame({"image_id": [img], "selected_idx": [json.dumps(idx)], "config_name": name,
                      "experiment_id": f"{name}@t", "code_version": "v1"}).to_csv(p, index=False)
    return cache, split, tmp_path, run


def test_identical_masks_share_item_but_keep_all_provenance(setup):
    cache, split, tmp, run = setup
    run("A", [0, 1])
    run("B", [2])
    items, prov = build_items(str(tmp / "selector_*_per_image.csv"), cache, split)
    assert set(items.review_id) == {"D7-c0", "D7-c1"}  # c2 merged into c0, c1 (similar) kept separate
    assert items.mask_sha256.is_unique
    p = prov[prov.review_id == "D7-c0"]
    assert set(p.candidate_id) == {"not_damaged_7#0", "not_damaged_7#2"} and set(p.config_name) == {"A", "B"}
    assert len(prov) == 3 and set(prov.columns) >= {"experiment_id", "code_version", "mask_sha256"}


def test_review_ids_are_stable_when_configs_are_added(setup):
    cache, split, tmp, run = setup
    run("A", [1])
    a, _ = build_items(str(tmp / "selector_*_per_image.csv"), cache, split)
    run("B", [0])
    b, _ = build_items(str(tmp / "selector_*_per_image.csv"), cache, split)
    assert "D7-c1" in set(a.review_id) and "D7-c1" in set(b.review_id)


def test_eval_images_rejected_and_no_runs_error(setup):
    cache, split, tmp, run = setup
    run("E", [0], img="not_damaged_9")
    with pytest.raises(ValueError, match="non-dev"):
        build_items(str(tmp / "selector_*_per_image.csv"), cache, split)
    with pytest.raises(FileNotFoundError):
        build_items(str(tmp / "nothing_*.csv"), cache, split)


def test_mask_hash_deterministic_and_discriminating():
    assert mask_sha256(_m(1, 5, 1, 5)) == mask_sha256(_m(1, 5, 1, 5).copy())
    assert mask_sha256(_m(1, 5, 1, 5)) != mask_sha256(_m(1, 5, 1, 6))


def test_render_panel_draws_something(setup):
    cache, split, tmp, run = setup
    run("A", [0])
    items, _ = build_items(str(tmp / "selector_*_per_image.csv"), cache, split)
    cv2.imwrite(str(tmp / "a.jpg"), np.zeros((H, W, 3), np.uint8))
    panel = render_panel("not_damaged_7", items, cache, tmp, "a.jpg")
    assert panel.shape == (H, W, 3) and panel.any()


def test_verdict_file_never_overwritten_and_append_only(setup):
    from src.experiments.review_pool import update_verdict_template

    cache, split, tmp, run = setup
    run("A", [1])
    items, _ = build_items(str(tmp / "selector_*_per_image.csv"), cache, split)
    vp = tmp / "review" / "verdicts.csv"
    assert update_verdict_template(vp, items) == 1
    d = pd.read_csv(vp, dtype=str, keep_default_na=False)
    assert (d.verdict == "").all()  # blank until a human/reviewer fills it
    d.loc[0, ["verdict", "uncertainty", "note"]] = ["egg", "low", "kept by hand"]
    d.to_csv(vp, index=False)
    assert update_verdict_template(vp, items) == 0  # idempotent
    run("B", [0])
    items2, _ = build_items(str(tmp / "selector_*_per_image.csv"), cache, split)
    assert update_verdict_template(vp, items2) == 1
    out = pd.read_csv(vp, dtype=str, keep_default_na=False).set_index("review_id")
    assert out.loc["D7-c1", "verdict"] == "egg" and out.loc["D7-c1", "note"] == "kept by hand"
    assert out.loc["D7-c0", "verdict"] == ""


def test_verdict_file_with_wrong_columns_is_refused(setup):
    from src.experiments.review_pool import update_verdict_template

    cache, split, tmp, run = setup
    run("A", [1])
    items, _ = build_items(str(tmp / "selector_*_per_image.csv"), cache, split)
    vp = tmp / "v.csv"
    vp.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="refusing"):
        update_verdict_template(vp, items)
    assert vp.read_text() == "foo,bar\n1,2\n"


def test_build_items_is_deterministic(setup):
    cache, split, tmp, run = setup
    run("A", [0, 1])
    run("B", [2, 1])
    a = build_items(str(tmp / "selector_*_per_image.csv"), cache, split)
    b = build_items(str(tmp / "selector_*_per_image.csv"), cache, split)
    assert a[0].equals(b[0]) and a[1].equals(b[1])
