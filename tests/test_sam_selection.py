import cv2
import numpy as np
import pytest

from src.segmentation.sam_segmenter import SamSegmenter, SelectionConfig, select_candidates, union_mask

H = W = 200


def ellipse(cx, cy, a, b, ang=0):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, ((cx, cy), (2 * a, 2 * b), ang), 1, -1)
    return m


def cand(mask, piou=0.95, stab=0.95):
    return {"mask": mask, "predicted_iou": piou, "stability_score": stab}


CFG = SelectionConfig()


def test_no_candidates_gives_empty_union():
    assert select_candidates([], (H, W), CFG) == []
    assert union_mask([], (H, W)).sum() == 0


def test_two_distinct_eggs_are_both_kept():
    sel = select_candidates([cand(ellipse(50, 50, 25, 18)), cand(ellipse(150, 150, 25, 18))], (H, W), CFG)
    assert len(sel) == 2
    assert union_mask(sel, (H, W)).sum() == sum(s["mask"].sum() for s in sel)


def test_duplicate_and_nested_masks_suppressed():
    big = ellipse(100, 100, 40, 30)
    near = ellipse(101, 100, 40, 30)
    inner = ellipse(100, 100, 20, 14)
    sel = select_candidates([cand(big, 0.97), cand(near, 0.9), cand(inner, 0.99)], (H, W), CFG)
    assert len(sel) == 1


def test_rejects_non_egg_shapes_and_low_confidence():
    bar = np.zeros((H, W), np.uint8)
    bar[90:100, 10:190] = 1  # elongated
    tiny = ellipse(20, 20, 2, 2)
    huge = np.ones((H, W), np.uint8)
    ell = ellipse(100, 100, 30, 22)
    assert select_candidates([cand(bar), cand(tiny), cand(huge)], (H, W), CFG) == []
    assert select_candidates([cand(ell, piou=0.5)], (H, W), CFG) == []
    assert select_candidates([cand(ell, stab=0.5)], (H, W), CFG) == []


def test_egg_clipped_by_image_border_is_kept():
    sel = select_candidates([cand(ellipse(10, 100, 30, 22))], (H, W), CFG)
    assert len(sel) == 1 and sel[0]["features"]["touches_border"]


def test_config_validation():
    with pytest.raises(ValueError):
        SelectionConfig(min_area_frac=0.9, max_area_frac=0.1).validate()
    with pytest.raises(ValueError):
        SelectionConfig(max_aspect=0.5).validate()


def test_missing_checkpoint_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        SamSegmenter(tmp_path / "nope.pth")


def test_fortran_ordered_mask_is_handled():
    """SAM returns non-C-contiguous masks; cv2.ellipse rejects those layouts (regression)."""
    m = np.asfortranarray(ellipse(100, 100, 30, 22))
    assert not m.flags["C_CONTIGUOUS"]
    sel = select_candidates([cand(m)], (H, W), CFG)
    assert len(sel) == 1
