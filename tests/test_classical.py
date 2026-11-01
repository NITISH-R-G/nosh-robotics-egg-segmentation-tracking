import hashlib

import cv2
import numpy as np
import pytest

from src.evaluation.metrics import mask_iou
from src.segmentation.classical import ClassicalConfig, foreground_map, segment_classical

H, W = 200, 300


def _scene(bg=(40, 40, 40), eggs=((100, 100, 40, 28),), egg=(235, 235, 235)):
    img = np.full((H, W, 3), bg, np.uint8)
    truth = []
    for cx, cy, a, b in eggs:
        cv2.ellipse(img, ((cx, cy), (2 * a, 2 * b), 0), egg, -1)
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, ((cx, cy), (2 * a, 2 * b), 0), 1, -1)
        truth.append(m.astype(bool))
    return img, truth


def test_uniform_and_blank_images_give_empty_mask_without_crashing():
    for v in (0, 128, 255):
        u, inst, info = segment_classical(np.full((H, W, 3), v, np.uint8), ClassicalConfig())
        assert not u.any() and inst == [] and info["kept"] == 0


def test_single_bright_egg_on_dark_background():
    img, (t,) = _scene()
    u, inst, _ = segment_classical(img, ClassicalConfig())
    assert len(inst) == 1 and mask_iou(u, t) > 0.95


def test_multiple_eggs_are_separate_instances_in_deterministic_order():
    img, truth = _scene(eggs=((60, 60, 28, 20), (200, 120, 30, 22), (90, 160, 25, 18)))
    u, inst, _ = segment_classical(img, ClassicalConfig())
    assert len(inst) == 3
    tops = [np.nonzero(m)[0].min() for m in inst]
    assert tops == sorted(tops)  # ordered by (top, left)
    for t in truth:
        assert max(mask_iou(m, t) for m in inst) > 0.9


def test_small_noisy_regions_are_removed_by_morphology_and_area_filter():
    img, (t,) = _scene()
    rng = np.random.default_rng(0)
    for _ in range(40):  # bright specks of 1-3 px
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        if abs(x - 100) > 60 or abs(y - 100) > 45:
            cv2.circle(img, (x, y), int(rng.integers(1, 3)), (255, 255, 255), -1)
    u, inst, _ = segment_classical(img, ClassicalConfig())
    assert len(inst) == 1 and mask_iou(u, t) > 0.9
    # a blob below the minimum area but too big for the opening kernel is rejected by the area filter alone
    img2, _ = _scene(eggs=((60, 60, 5, 5), (200, 120, 35, 25)))
    _, inst2, info = segment_classical(img2, ClassicalConfig(open_k=0, close_k=0, min_area_frac=0.01))
    assert len(inst2) == 1 and info["rejected"]["area"] >= 1


def test_touching_eggs_merge_by_default_and_split_with_watershed():
    img, truth = _scene(eggs=((90, 100, 40, 28), (160, 100, 40, 28)))  # centres 70 apart, radii 40 -> overlap
    u, inst, _ = segment_classical(img, ClassicalConfig())
    assert len(inst) == 1  # one merged component: documented limitation without the split step
    u2, inst2, _ = segment_classical(img, ClassicalConfig(split_watershed=True, split_h=0.4))  # neck prominence is ~0.45 of the peak
    assert len(inst2) == 2 and abs(int(inst2[0].sum()) - int(inst2[1].sum())) < 0.1 * int(inst2[0].sum())  # balanced split
    assert len(segment_classical(img, ClassicalConfig(split_watershed=True, split_h=0.6))[1]) == 1  # a higher bar keeps them merged
    for t in truth:
        assert max(mask_iou(m, t) for m in inst2) > 0.7


def test_white_on_white_is_a_documented_failure_of_the_lightness_baseline():
    good, (tg,) = _scene(bg=(40, 40, 40))
    bad, (tb,) = _scene(bg=(225, 225, 225), egg=(238, 238, 238))
    # realistic conditions: sensor noise and a lighting gradient across the frame (a noise-free two-level image is trivially separable)
    rng = np.random.default_rng(0)
    bad = np.clip(bad.astype(float) + np.linspace(-25, 15, W)[None, :, None] + rng.normal(0, 7, bad.shape), 0, 255).astype(np.uint8)
    ug, _, _ = segment_classical(good, ClassicalConfig())
    ub, instb, _ = segment_classical(bad, ClassicalConfig())
    assert mask_iou(ug, tg) > 0.95
    assert mask_iou(ub, tb) < 0.5  # the egg is not separable from a near-identical, unevenly lit background


def test_backlit_dark_egg_on_bright_background_is_missed_by_lightness_baseline():
    img, (t,) = _scene(bg=(230, 230, 230), egg=(60, 60, 60))
    u, inst, _ = segment_classical(img, ClassicalConfig())
    assert mask_iou(u, t) < 0.2  # the baseline assumes eggs are brighter than their surroundings


def test_deterministic_and_independent_of_memory_layout():
    img, _ = _scene(eggs=((60, 60, 28, 20), (200, 120, 30, 22)))
    h = lambda r: hashlib.sha256(r[0].tobytes()).hexdigest()  # noqa: E731
    cfg = ClassicalConfig(colour="W", blur_sigma=2.0, split_watershed=True)
    a, b = segment_classical(img, cfg), segment_classical(img.copy(), cfg)
    f = segment_classical(np.asfortranarray(img), cfg)
    assert h(a) == h(b) == h(f)


def test_filters_remove_non_egg_shapes_when_enabled_only():
    img = np.full((H, W, 3), 40, np.uint8)
    cv2.rectangle(img, (20, 90), (280, 110), (235, 235, 235), -1)      # long thin bar: aspect 13
    pts = np.array([[200, 20], [260, 20], [260, 40], [215, 40], [215, 80], [200, 80]], np.int32)
    cv2.fillPoly(img, [pts], (235, 235, 235))                           # L-shape: low solidity
    cv2.ellipse(img, ((80, 150), (80, 56), 0), (235, 235, 235), -1)     # the only egg-like shape
    base = segment_classical(img, ClassicalConfig())[1]
    assert len(base) == 3
    assert len(segment_classical(img, ClassicalConfig(max_aspect=2.2))[1]) == 2
    assert len(segment_classical(img, ClassicalConfig(min_solidity=0.85))[1]) == 2
    # a thin bar fits an ellipse reasonably (IoU ~0.86): the 0.85 gate used in experiment E13 lets it pass, the aspect filter is what removes it
    assert len(segment_classical(img, ClassicalConfig(min_ellipse_iou=0.85))[1]) == 2
    assert len(segment_classical(img, ClassicalConfig(min_ellipse_iou=0.9))[1]) == 1
    assert len(segment_classical(img, ClassicalConfig(max_aspect=2.2, min_solidity=0.85, min_ellipse_iou=0.85))[1]) == 1


def test_border_rule_drops_small_border_components_only():
    img = np.full((H, W, 3), 40, np.uint8)
    cv2.ellipse(img, ((0, 100), (40, 40), 0), (235, 235, 235), -1)      # small blob cut by the left border
    cv2.ellipse(img, ((200, 100), (60, 44), 0), (235, 235, 235), -1)    # interior egg
    assert len(segment_classical(img, ClassicalConfig())[1]) == 2
    kept = segment_classical(img, ClassicalConfig(border_min_area_frac=0.05))[1]
    assert len(kept) == 1 and np.nonzero(kept[0])[1].min() > 100


def test_colour_maps_orient_eggs_high_and_coloured_regions_low():
    img = np.zeros((10, 20, 3), np.uint8)
    img[:, :10] = (235, 235, 235)   # white
    img[:, 10:] = (60, 60, 230)     # saturated red
    for c in "LVWS":
        f = foreground_map(img, c)
        assert f[:, :10].mean() > f[:, 10:].mean(), c


def test_invalid_config_and_image_rejected():
    for bad in (ClassicalConfig(colour="X"), ClassicalConfig(adaptive_block=100), ClassicalConfig(min_area_frac=0.7),
                ClassicalConfig(max_aspect=0.5), ClassicalConfig(split_h=1.5), ClassicalConfig(min_solidity=2.0)):
        with pytest.raises(ValueError):
            bad.validate()
    with pytest.raises(ValueError):
        segment_classical(np.zeros((10, 10), np.uint8), ClassicalConfig())
    with pytest.raises(ValueError):
        segment_classical(np.zeros((10, 10, 3), np.float32), ClassicalConfig())
    assert ClassicalConfig().sha() == ClassicalConfig().sha() != ClassicalConfig(open_k=3).sha()
