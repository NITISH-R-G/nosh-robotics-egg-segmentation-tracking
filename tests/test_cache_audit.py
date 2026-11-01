import json

import cv2
import numpy as np
import pytest

from src.segmentation.audit_cache import audit
from src.segmentation.cache_candidates import FORMAT_VERSION, candidate_record, pack_masks, unpack_masks

H, W = 21, 30  # not a multiple of 8 in H*W (630 bits) to exercise padding


def _mask(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def _cand(m):
    ys, xs = np.nonzero(m)
    return {"bbox_xywh": [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1],
            "predicted_iou": 0.9, "stability_score": 0.9, "point_coords": [1.0, 1.0], "crop_box": [0, 0, W, H]}


@pytest.fixture
def cache(tmp_path):
    data, out = tmp_path / "data", tmp_path / "cache"
    (data / "train").mkdir(parents=True)
    (out / "masks").mkdir(parents=True)
    rows = []
    for img_id, masks in (("a", [_mask(2, 8, 3, 9), _mask(10, 20, 15, 29)]), ("b", [])):
        cv2.imwrite(str(data / "train" / f"{img_id}_jpg.rf.abc.jpg"), np.zeros((H, W, 3), np.uint8))
        if masks:
            packed, shape = pack_masks(masks)
        else:
            packed, shape = np.zeros((0, 0), np.uint8), (H, W)
        np.savez_compressed(out / "masks" / f"{img_id}.npz", masks=packed, shape=np.array(shape))
        rows += [candidate_record(img_id, k, f"masks/{img_id}.npz", m, _cand(m)) for k, m in enumerate(masks)]
    (out / "candidates.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "run.json").write_text(json.dumps({"format_version": FORMAT_VERSION, "total_runtime_s": 1.0}))
    return data, out


def test_clean_cache_passes_with_zero_candidate_image(cache):
    res = audit(cache[1], cache[0])
    assert res["ok"], res["errors"]
    assert res["stats"]["empty_candidate_images"] == ["b"] and res["stats"]["n_masks_verified"] == 2


def test_reconstruction_is_exact(cache):
    out = unpack_masks(cache[1] / "masks" / "a.npz")
    assert out.shape == (2, H, W) and (out[0] == _mask(2, 8, 3, 9)).all() and (out[1] == _mask(10, 20, 15, 29)).all()


def test_corrupt_npz_detected(cache):
    p = cache[1] / "masks" / "a.npz"
    p.write_bytes(p.read_bytes()[:40])
    res = audit(cache[1], cache[0])
    assert not res["ok"] and any("unreadable npz" in e for e in res["errors"])


def test_wrong_shape_detected(cache):
    p = cache[1] / "masks" / "a.npz"
    z = np.load(p)
    np.savez_compressed(p, masks=z["masks"], shape=np.array((H + 1, W)))
    assert any("npz shape" in e for e in audit(cache[1], cache[0])["errors"])


def test_manifest_count_mismatch_and_duplicates_detected(cache):
    jl = cache[1] / "candidates.jsonl"
    lines = jl.read_text().splitlines()
    jl.write_text("\n".join(lines[:1]) + "\n")
    assert any("manifest has 1 rows" in e for e in audit(cache[1], cache[0])["errors"])
    jl.write_text("\n".join([lines[0], lines[0]]) + "\n")
    assert any("duplicate candidate_ids" in e for e in audit(cache[1], cache[0])["errors"])


def test_missing_npz_and_area_tamper_detected(cache):
    (cache[1] / "masks" / "b.npz").unlink()
    assert any("without npz" in e for e in audit(cache[1], cache[0])["errors"])
    jl = cache[1] / "candidates.jsonl"
    recs = [json.loads(line) for line in jl.read_text().splitlines()]
    recs[0]["area"] += 1
    jl.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert any("area" in e for e in audit(cache[1], cache[0])["errors"])


def test_nan_feature_detected(cache):
    jl = cache[1] / "candidates.jsonl"
    recs = [json.loads(line) for line in jl.read_text().splitlines()]
    recs[0]["ellipse_iou"] = None
    jl.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert any("non-finite" in e for e in audit(cache[1], cache[0])["errors"])
