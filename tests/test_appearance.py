import numpy as np
import pytest

from src.analysis.appearance import auc, best_threshold, cluster_bootstrap_auc, descriptors, loso_threshold

H, W = 120, 160


def _scene(egg_bgr=(235, 235, 235), bg_bgr=(40, 40, 40)):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = bg_bgr
    mask = np.zeros((H, W), bool)
    yy, xx = np.mgrid[:H, :W]
    mask[((xx - 100) / 30) ** 2 + ((yy - 60) / 22) ** 2 <= 1] = True
    img[mask] = egg_bgr
    return img, mask


def test_bright_unsaturated_smooth_object_on_dark_background():
    img, mask = _scene()
    d = descriptors(img, mask)
    assert d["L_in"] > 85 and d["chroma_in"] < 3 and d["dL_ring"] > 60
    assert d["texture_in"] < 1.0 and d["edge_strength"] > 20  # flat interior, strong boundary


def test_coloured_textured_region_scores_less_egg_like():
    img, mask = _scene(egg_bgr=(80, 170, 230))  # saturated orange-ish object
    rng = np.random.default_rng(0)
    img[mask] = np.clip(img[mask].astype(int) + rng.integers(-40, 40, img[mask].shape), 0, 255)
    d_noisy, d_egg = descriptors(img, mask), descriptors(*_scene())
    assert d_noisy["chroma_in"] > d_egg["chroma_in"] + 20 and d_noisy["texture_in"] > d_egg["texture_in"] + 5


def test_white_on_white_has_weak_contrast_and_edge():
    weak = descriptors(*_scene(egg_bgr=(235, 235, 235), bg_bgr=(225, 225, 225)))
    strong = descriptors(*_scene())
    assert abs(weak["dL_ring"]) < 8 < strong["dL_ring"]
    assert weak["edge_strength"] < 0.1 * strong["edge_strength"]  # the documented failure mode of H_dL / H_edge


def test_descriptor_input_validation_and_border_masks():
    img, mask = _scene()
    with pytest.raises(ValueError):
        descriptors(img, mask[:-1])
    with pytest.raises(ValueError):
        descriptors(img, np.zeros_like(mask))
    d = descriptors(img, np.ones((H, W), bool))  # mask filling the frame: no outside ring exists, must not crash
    assert np.isnan(d["dL_ring"]) and all(np.isfinite(v) for k, v in d.items() if k != "dL_ring")


def test_auc_known_values_and_ties():
    assert auc([3, 4, 5], [0, 1, 2]) == 1.0 and auc([0, 1], [3, 4]) == 0.0
    assert auc([1, 1], [1, 1]) == 0.5
    assert auc([2, 3], [1, 3]) == pytest.approx(0.625)  # pairs: 2>1, 2<3, 3>1, 3=3 -> (1 + 0 + 1 + 0.5) / 4
    assert np.isnan(auc([], [1]))


def test_best_threshold_separates_clean_classes():
    s, y = np.array([1, 2, 3, 10, 11, 12.0]), np.array([0, 0, 0, 1, 1, 1])
    t = best_threshold(s, y)
    assert 3 < t <= 10 and ((s >= t) == (y == 1)).all()


def test_loso_never_uses_the_held_out_group_to_choose_its_threshold():
    s = np.array([1, 2, 10, 11, 1, 2, 10, 11, 10, 11, 1, 2.0])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])  # group C has the OPPOSITE relation
    g = np.array(list("AAAABBBBCCCC"))
    p = loso_threshold(s, y, g)
    for grp in "ABC":  # each group's prediction equals thresholding with a cut-off fitted on the OTHER groups only
        te = g == grp
        assert (p[te] == (s[te] >= best_threshold(s[~te], y[~te])).astype(int)).all()
    # a group's own labels must not change its own predictions
    y2 = y.copy()
    y2[g == "C"] = 1 - y2[g == "C"]
    assert (loso_threshold(s, y2, g)[g == "C"] == p[g == "C"]).all()
    # sanity: with consistent groups the held-out group is predicted correctly
    y_ok = np.array([0, 0, 1, 1] * 3)
    s_ok = np.array([1, 2, 10, 11] * 3, dtype=float)
    assert (loso_threshold(s_ok, y_ok, g) == y_ok).all()


def test_cluster_bootstrap_interval_contains_point_estimate_for_separable_data():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(8), 10)
    y = np.tile([1, 0], 40)
    s = y * 2.0 + rng.normal(0, 0.5, 80)
    lo, hi = cluster_bootstrap_auc(s, y, g, n_boot=200)
    assert lo <= auc(s[y == 1], s[y == 0]) <= hi + 1e-9 and lo > 0.8
