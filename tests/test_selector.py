import numpy as np
import pandas as pd
import pytest

from src.segmentation.selector import SelectorConfig, select_from_features

H = W = 100
AREA = H * W


def row(area, e=0.97, p=0.97, s=0.97, sol=0.98, asp=1.4, border=False):
    return {"area": area, "ellipse_iou": e, "predicted_iou": p, "stability_score": s, "solidity": sol,
            "aspect": asp, "touches_border": border}


def run(rows, masks=None, cfg=None):
    df = pd.DataFrame(rows)
    masks = masks or {i: np.zeros((H, W), bool) for i in df.index}
    for i in df.index:  # disjoint dummy masks unless provided
        if i not in masks:
            m = np.zeros((H, W), bool); m[i : i + 1, :5] = True; masks[i] = m
    return select_from_features(df, lambda i: masks[i], AREA, cfg or SelectorConfig("t"))


def test_basic_pass_and_reject():
    sel, st = run([row(1500), row(1500, e=0.5), row(20), row(9000)])
    assert sel == [0] and st == "ok"  # low ellipse fit, tiny area, huge area rejected


def test_empty_candidates():
    sel, st = run([])
    assert sel == [] and st == "none_selected"


def test_nan_aspect_rejected():
    sel, _ = run([row(1500, asp=np.nan)])
    assert sel == []


def test_border_rule_relaxes_ellipse_only_for_large_clipped():
    cfg = SelectorConfig("b", min_ellipse_iou=0.95, border_min_ellipse_iou=0.85, border_min_area_frac=0.05)
    big = [row(800, e=0.88, border=True)]
    assert run(big, cfg=cfg)[0] == [0]  # 0.08 area frac passes the border rule
    small = [row(300, e=0.88, border=True)]
    assert run(small, cfg=cfg)[0] == []  # below border_min_area_frac
    assert run([row(800, e=0.88, border=False)], cfg=cfg)[0] == []  # non-border keeps strict threshold


def test_nms_suppresses_overlap_keeps_disjoint():
    a = np.zeros((H, W), bool); a[10:40, 10:40] = True
    b = np.zeros((H, W), bool); b[11:40, 10:40] = True
    c = np.zeros((H, W), bool); c[60:90, 60:90] = True
    sel, _ = run([row(900, e=0.99), row(899, e=0.97), row(900, e=0.96)], masks={0: a, 1: b, 2: c})
    assert sel == [0, 2]


def test_fallbacks():
    weak = [row(1500, e=0.90, p=0.85)]
    assert run(weak, cfg=SelectorConfig("n", min_ellipse_iou=0.95))[1] == "none_selected"
    sel, st = run(weak, cfg=SelectorConfig("r", min_ellipse_iou=0.95, fallback="review"))
    assert sel == [] and st == "review_no_candidate"
    sel, st = run(weak, cfg=SelectorConfig("f", min_ellipse_iou=0.95, fallback="relaxed_top1"))
    assert sel == [0] and st == "fallback_top1"
    bad = [row(1500, e=0.5)]
    assert run(bad, cfg=SelectorConfig("f", fallback="relaxed_top1"))[1] == "review_no_candidate"


def test_bad_config():
    with pytest.raises(ValueError):
        SelectorConfig("x", fallback="magic").validate()


# ---------------- alt gates and relative-area rule ----------------
S1_LIKE = dict(min_area_frac=0.01, max_area_frac=0.60, min_solidity=0.95, min_ellipse_iou=0.93, min_pred_iou=0.90,
               min_stability=0.90)
SHAPE_ALT = SelectorConfig("alt", min_area_frac=0.02, max_area_frac=0.60, min_solidity=0.78, min_ellipse_iou=0.72,
                           min_pred_iou=0.95, min_stability=0.93)
CONF_ALT = SelectorConfig("alt", min_area_frac=0.02, max_area_frac=0.10, min_solidity=0.88, min_ellipse_iou=0.89,
                          min_pred_iou=0.65, min_stability=0.55)


def test_shape_alt_admits_confident_occluded_but_not_unconfident_or_tiny():
    base = SelectorConfig("b", **S1_LIKE)
    cfg = SelectorConfig("e1", **S1_LIKE, alts=(SHAPE_ALT,))
    occluded = row(2500, e=0.78, p=0.985, s=0.94, sol=0.84)
    assert run([occluded], cfg=base)[0] == [] and run([occluded], cfg=cfg)[0] == [0]
    assert run([row(2500, e=0.78, p=0.80, s=0.94, sol=0.84)], cfg=cfg)[0] == []   # not confident
    assert run([row(100, e=0.78, p=0.985, s=0.94, sol=0.84)], cfg=cfg)[0] == []   # below alt area floor
    assert run([row(2500, e=0.60, p=0.985, s=0.94, sol=0.84)], cfg=cfg)[0] == []  # shape too poor


def test_conf_alt_admits_low_confidence_good_shape_only_in_area_band():
    base = SelectorConfig("b", **S1_LIKE)
    cfg = SelectorConfig("e2", **S1_LIKE, alts=(CONF_ALT,))
    sub = row(500, e=0.90, p=0.67, s=0.58, sol=0.95)  # 5% of the 100x100 test image
    assert run([sub], cfg=base)[0] == [] and run([sub], cfg=cfg)[0] == [0]
    assert run([row(2000, e=0.90, p=0.67, s=0.58, sol=0.95)], cfg=cfg)[0] == []   # 20% > alt max area
    assert run([row(500, e=0.85, p=0.67, s=0.58, sol=0.95)], cfg=cfg)[0] == []    # shape too poor


def test_alts_do_not_change_selection_when_empty_and_tier1_wins_nms():
    base = SelectorConfig("b", **S1_LIKE)
    same = SelectorConfig("b2", **S1_LIKE, alts=())
    rows = [row(1500), row(1500, e=0.5)]
    assert run(rows, cfg=base) == run(rows, cfg=same)
    a = np.zeros((H, W), bool); a[10:40, 10:40] = True
    b = np.zeros((H, W), bool); b[11:41, 10:40] = True  # overlaps a
    cfg = SelectorConfig("e1", **S1_LIKE, alts=(SHAPE_ALT,))
    sel, _ = run([row(900, e=0.99), row(900, e=0.78, p=0.985, s=0.94, sol=0.84)], masks={0: a, 1: b}, cfg=cfg)
    assert sel == [0]  # relaxed duplicate suppressed by the stronger tier-1 candidate


def test_rel_area_drops_small_instances_only_when_several_selected():
    cfg = SelectorConfig("e4", **S1_LIKE, min_rel_area=0.25)
    sel, _ = run([row(5000), row(600)])  # sanity with default cfg keeps both
    assert sel == [0, 1]
    sel, st = run([row(5000), row(600)], cfg=cfg)
    assert sel == [0] and st == "ok"
    sel, _ = run([row(5000), row(1500)], cfg=cfg)  # ratio 0.3 kept
    assert sel == [0, 1]
    assert run([row(600)], cfg=cfg)[0] == [0]      # a lone small egg is never dropped


def test_bad_rel_area_and_nested_alt_validation():
    with pytest.raises(ValueError):
        SelectorConfig("x", min_rel_area=1.5).validate()
    with pytest.raises(ValueError):
        SelectorConfig("x", alts=(SelectorConfig("bad", fallback="magic"),)).validate()


def test_alt_gate_fallback_only_applies_when_primary_selects_nothing():
    cfg = SelectorConfig("e5", **{**S1_LIKE, "fallback": "alt_gate"}, fallback_alts=(CONF_ALT,))
    sub = row(500, e=0.90, p=0.67, s=0.58, sol=0.95)
    sel, st = run([sub], cfg=cfg)
    assert sel == [0] and st == "alt_gate_fallback"
    sel, st = run([row(1500), sub], cfg=cfg)  # primary finds an egg -> relaxed candidate must NOT be added
    assert sel == [0] and st == "ok"
    assert run([row(1500, e=0.5)], cfg=cfg) == ([], "review_no_candidate")
    assert run([], cfg=cfg) == ([], "review_no_candidate")
    with pytest.raises(ValueError):
        SelectorConfig("x", fallback="alt_gate").validate()
