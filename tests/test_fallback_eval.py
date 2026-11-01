import cv2
import numpy as np
import pandas as pd
import pytest

from src.experiments.fallback_eval import (CRITERIA, SCORE_FEATURES, build_pool, config_metrics, evaluate,
                                           mask_overlap_duplicates, summarize, z_score)
from src.segmentation.cache_candidates import candidate_record, pack_masks

H = W = 100


def _ell(cx, cy, a, b):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, ((cx, cy), (2 * a, 2 * b), 0), 1, -1)
    return m.astype(bool)


def _table(rows):
    """rows: (image, review_id, mask_sha, verdict, L_in, dL_ring, texture_in)"""
    return pd.DataFrame(rows, columns=["image_id", "review_id", "mask_sha256", "verdict", "L_in", "dL_ring", "texture_in"]).assign(
        candidate_index=0)


def test_z_score_orientation_texture_low_is_more_egg_like():
    mu = pd.Series({"L_in": 50.0, "dL_ring": 0.0, "texture_in": 10.0})
    sd = pd.Series({"L_in": 10.0, "dL_ring": 10.0, "texture_in": 5.0})
    egg = pd.DataFrame({"L_in": [70.0], "dL_ring": [20.0], "texture_in": [0.0]})
    bg = pd.DataFrame({"L_in": [30.0], "dL_ring": [-20.0], "texture_in": [20.0]})
    assert z_score(egg, mu, sd)[0] > 0 > z_score(bg, mu, sd)[0]
    assert SCORE_FEATURES == {"L_in": 1, "dL_ring": 1, "texture_in": -1}


def test_evaluate_never_uses_the_held_out_image_for_statistics_or_threshold():
    rows = []
    for k in range(6):  # six images, each with one egg-like and one non-egg pool candidate
        rows += [(f"i{k}", f"R{k}a", f"h{k}a", "egg", 80.0 + k, 20.0, 1.0), (f"i{k}", f"R{k}b", f"h{k}b", "not_egg", 30.0 + k, -10.0, 15.0)]
    t = _table(rows)
    pool = t[["image_id", "mask_sha256"]].assign(candidate_index=0, candidate_id=t.image_id + "#0", pool_status="alt_gate_fallback")
    base = evaluate(pool, t)
    # add an extreme reviewed candidate to image i0 that is NOT in the pool: it must not change i0's own score or threshold
    extra = _table([("i0", "RX", "hx", "not_egg", 1e6, 1e6, -1e6)])
    leaked = evaluate(pool, pd.concat([t, extra], ignore_index=True))
    a, b = base[base.image_id == "i0"].reset_index(drop=True), leaked[leaked.image_id == "i0"].reset_index(drop=True)
    assert np.allclose(a.score, b.score) and np.allclose(a.tau_from_other_images, b.tau_from_other_images)
    # but the same extreme candidate in ANOTHER image does change i0's statistics (the held-out rule is really out-of-image)
    extra2 = _table([("i3", "RX", "hx", "not_egg", 1e6, 1e6, -1e6)])
    other = evaluate(pool, pd.concat([t, extra2], ignore_index=True))
    assert not np.allclose(a.score, other[other.image_id == "i0"].reset_index(drop=True).score)


def test_unreviewed_pool_candidates_raise():
    t = _table([("i0", "R0", "h0", "egg", 80.0, 20.0, 1.0)])
    pool = pd.DataFrame({"image_id": ["i0"], "candidate_index": [3], "candidate_id": ["i0#3"], "mask_sha256": ["unknown"], "pool_status": "x"})
    with pytest.raises(ValueError, match="unreviewed"):
        evaluate(pool, t)


def test_build_pool_skips_images_where_s1_selected_and_uses_relaxed_gates(tmp_path):
    cache = tmp_path / "c"
    (cache / "masks").mkdir(parents=True)
    ms = {"a": [_ell(50, 50, 20, 14)], "b": [_ell(50, 50, 20, 14)]}
    lines = []
    for img, masks in ms.items():
        packed, shape = pack_masks(masks)
        np.savez_compressed(cache / "masks" / f"{img}.npz", masks=packed, shape=np.array(shape))
        for k, m in enumerate(masks):
            ys, xs = np.nonzero(m)
            # confidence too low for S1 but within the alt_conf gate
            c = {"bbox_xywh": [xs.min(), ys.min(), 1, 1], "predicted_iou": 0.70, "stability_score": 0.60, "point_coords": [1.0, 1.0],
                 "crop_box": [0, 0, W, H]}
            lines.append(candidate_record(img, k, f"masks/{img}.npz", m, c))
    import json

    (cache / "candidates.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    from src.segmentation.cache_candidates import read_candidates

    cands = read_candidates(cache, {"a", "b"})
    pool = build_pool(cands, ["a", "b"], cache, {"a": [0], "b": []})  # S1 selected something in a only
    assert list(pool.image_id) == ["b"] and pool.pool_status.iloc[0] == "alt_gate_fallback" and len(pool.mask_sha256[0]) == 64


def test_config_metrics_counts_cancelling_free_and_duplicates(tmp_path):
    cache = tmp_path / "c"
    (cache / "masks").mkdir(parents=True)
    a, b = _ell(40, 40, 20, 14), _ell(42, 40, 20, 14)  # near-identical masks: a duplicate pair
    packed, shape = pack_masks([a, b, _ell(80, 80, 10, 8)])
    np.savez_compressed(cache / "masks" / "i1.npz", masks=packed, shape=np.array(shape))
    counts = pd.DataFrame({"image_id": ["i1", "i2"], "expected_egg_count": [1, 1]})
    sel = pd.DataFrame({"image_id": ["i1", "i1", "i1"], "candidate_id": ["i1#0", "i1#1", "i1#2"], "verdict": ["egg", "egg", "not_egg"]})
    assert mask_overlap_duplicates(cache, sel) == 1
    row, per = config_metrics("t", sel, counts, cache)
    assert (row["selected"], row["tp_lenient"], row["false_positives"], row["missed_eggs"]) == (3, 2, 1, 1)  # i2 selected nothing -> missed
    assert row["duplicate_pairs"] == 1 and row["count_exact_rate"] == 0.0 and row["max_abs_err"] == 2


def test_summarize_applies_preregistered_criteria():
    def res(verdicts, accepted):
        return pd.DataFrame({"verdict": verdicts, "accepted": accepted,
                             "egg_like": [int(v in ("egg", "egg_bad_boundary")) for v in verdicts]})
    ok = summarize(res(["egg"] * 5 + ["not_egg"], [True] * 6))
    assert ok["criteria_met"] and ok["new_egg_like"] == 5 and ok["new_false_positives"] == 1
    assert not summarize(res(["egg"] * 5 + ["not_egg"] * 2, [True] * 7))["criteria_met"]          # 2 false positives
    assert not summarize(res(["egg"] * 2 + ["egg_bad_boundary"] * 3, [True] * 5))["criteria_met"]  # clean share 0.4 < 0.6
    assert not summarize(res(["egg"] * 4, [True] * 4))["criteria_met"]                              # only 4 new eggs
    assert not summarize(res(["egg"] * 3, [False] * 3))["criteria_met"]                             # nothing accepted
    assert CRITERIA == {"min_new_egg_like": 5, "max_new_false_positives": 1, "min_clean_egg_share": 0.60}
