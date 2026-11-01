import math

import numpy as np
import pytest

from src.evaluation.metrics import (connected_instances, count_metrics, match_instances, mask_iou, pixel_metrics, prf)

H, W = 40, 60


def _rect(x0, x1, y0, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_identical_masks():
    m = _rect(5, 25, 5, 25)
    r = pixel_metrics(m, m)
    assert r["iou"] == 1.0 and r["dice"] == 1.0 and r["precision"] == 1.0 and r["recall"] == 1.0 and r["fp"] == 0 and r["fn"] == 0


def test_disjoint_masks():
    r = pixel_metrics(_rect(0, 10, 0, 10), _rect(30, 40, 20, 30))
    assert r["iou"] == 0.0 and r["dice"] == 0.0 and r["precision"] == 0.0 and r["recall"] == 0.0 and r["tp"] == 0


def test_partial_overlap_exact_values():
    ref, pred = _rect(0, 10, 0, 10), _rect(5, 15, 0, 10)  # 100 px each, 50 px overlap
    r = pixel_metrics(ref, pred)
    assert (r["tp"], r["fp"], r["fn"]) == (50, 50, 50)
    assert r["iou"] == pytest.approx(50 / 150) and r["dice"] == pytest.approx(0.5)
    assert r["precision"] == pytest.approx(0.5) and r["recall"] == pytest.approx(0.5)
    assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == H * W


def test_empty_reference_and_empty_prediction_are_explicit_not_silent():
    e, m = np.zeros((H, W), bool), _rect(5, 15, 5, 15)
    both = pixel_metrics(e, e)
    assert both["ref_empty"] and both["pred_empty"] and math.isnan(both["iou"]) and math.isnan(both["precision"])
    fp_only = pixel_metrics(e, m)
    assert fp_only["ref_empty"] and fp_only["fp"] == 100 and fp_only["precision"] == 0.0 and math.isnan(fp_only["iou"])
    miss = pixel_metrics(m, e)
    assert miss["iou"] == 0.0 and miss["dice"] == 0.0 and miss["recall"] == 0.0 and miss["pred_empty"] and math.isnan(miss["precision"])


def test_shape_mismatch_and_bad_dim_rejected():
    with pytest.raises(ValueError):
        pixel_metrics(np.zeros((3, 3), bool), np.zeros((3, 4), bool))
    with pytest.raises(ValueError):
        pixel_metrics(np.zeros((3, 3, 3), bool), np.zeros((3, 3, 3), bool))


def test_connected_instances_separates_blobs_and_ignores_small():
    m = _rect(2, 12, 2, 12) | _rect(30, 45, 10, 30)
    m[0, 59] = True  # single-pixel noise
    assert len(connected_instances(m)) == 3 and len(connected_instances(m, min_area=5)) == 2
    assert connected_instances(np.zeros((H, W), bool)) == []
    touching = _rect(0, 10, 0, 10) | _rect(10, 20, 0, 10)
    assert len(connected_instances(touching)) == 1  # touching eggs form ONE component


def test_matching_perfect_fp_miss_and_threshold():
    a, b = _rect(0, 10, 0, 10), _rect(30, 40, 20, 30)
    r = match_instances([a, b], [a, b])
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)
    r = match_instances([a, b, _rect(50, 58, 30, 38)], [a, b])
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 0) and r["fp_idx"] == [2]
    r = match_instances([a], [a, b])
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 1) and r["fn_idx"] == [1]
    shifted = _rect(6, 16, 0, 10)  # IoU with a = 40/160 = 0.25 < 0.5
    assert match_instances([shifted], [a])["tp"] == 0 and mask_iou(shifted, a) == pytest.approx(0.25)


def test_false_positive_cannot_hide_a_miss():
    a, b, c = _rect(0, 10, 0, 10), _rect(30, 40, 20, 30), _rect(50, 58, 30, 38)
    r = match_instances([a, c], [a, b])  # same count (2 vs 2) but one egg is wrong
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)


def test_merged_component_matches_only_one_of_two_touching_eggs():
    e1, e2 = _rect(0, 10, 0, 10), _rect(10, 20, 0, 10)
    merged = e1 | e2
    r = match_instances(connected_instances(merged), [e1, e2], 0.5)
    assert r["tp"] == 1 and r["fn"] == 1 and r["fp"] == 0  # IoU of merged with one egg is exactly 0.5
    r2 = match_instances(connected_instances(merged), [e1, e2], 0.51)
    assert r2["tp"] == 0 and r2["fp"] == 1 and r2["fn"] == 2


def test_prf_and_zero_division():
    assert prf(2, 1, 1) == pytest.approx({"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3})
    z = prf(0, 0, 0)
    assert all(math.isnan(v) for v in z.values())


def test_count_metrics():
    r = count_metrics(np.array([1, 2, 0, 3]), np.array([1, 1, 1, 3]))
    assert r["exact_rate"] == 0.5 and r["over_rate"] == 0.25 and r["under_rate"] == 0.25
    assert r["mean_abs_err"] == 0.5 and r["median_abs_err"] == 0.5 and r["max_abs_err"] == 1
    with pytest.raises(ValueError):
        count_metrics(np.array([1]), np.array([1, 2]))
