import cv2
import numpy as np
import pandas as pd
import pytest

from src.analysis.boundary_eval import (build_blind_panels, case_ids, classify, evaluate_criteria, record_ratings, resolve)
from src.segmentation.cache_candidates import pack_masks
from src.segmentation.masks import save_mask_png

H, W = 120, 160


def _ell(cx, cy, a, b):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, ((cx, cy), (2 * a, 2 * b), 0), 1, -1)
    return m.astype(bool)


def test_blind_panels_hide_identity_and_mapping_is_consistent(tmp_path):
    cache = tmp_path / "cache"
    (cache / "masks").mkdir(parents=True)
    orig, v1, v2 = _ell(60, 60, 25, 18), _ell(62, 60, 30, 20), _ell(60, 62, 22, 16)
    packed, shape = pack_masks([orig])
    np.savez_compressed(cache / "masks" / "img_1.npz", masks=packed, shape=np.array(shape))
    cv2.imwrite(str(tmp_path / "img_1.jpg"), np.full((H, W, 3), 90, np.uint8))
    ref = tmp_path / "ref"
    for name, m in (("V1", v1), ("V2", v2)):
        save_mask_png(ref / name / "masks" / "img_1__0.png", m)
    split = tmp_path / "split.csv"
    pd.DataFrame({"image_id": ["img_1"], "split": ["dev"], "path": ["img_1.jpg"]}).to_csv(split, index=False)
    cases = pd.DataFrame({"review_id": ["D1-c0"], "image_id": ["img_1"], "candidate_index": [0], "candidate_id": ["img_1#0"]})
    idx = build_blind_panels(cases, ref, cache, tmp_path, split, tmp_path / "out", seed=1)
    mp = pd.read_csv(tmp_path / "out" / "blind_mapping.csv")
    assert sorted(mp.loc[0, ["column_A", "column_B", "column_C"]]) == ["V1", "V2", "orig"]  # a permutation of the three masks
    sheet = cv2.imread(str(tmp_path / "out" / idx.sheet[0]))
    assert sheet.shape[1] == 3 * 480 and idx.case_id[0] == "C01"
    mp2 = build_blind_panels(cases, ref, cache, tmp_path, split, tmp_path / "out2", seed=1) is not None
    assert mp2 and pd.read_csv(tmp_path / "out2" / "blind_mapping.csv").equals(mp)  # seeded: reproducible
    bad = cases.assign(image_id="img_9")
    with pytest.raises(ValueError, match="non-development"):
        build_blind_panels(bad, ref, cache, tmp_path, split, tmp_path / "o3")


def test_rating_capture_validates_and_is_append_only(tmp_path):
    p = tmp_path / "r.csv"
    ok = {("C01", "A"): (3, "yes", "none", "plausible", "good")}
    assert record_ratings(p, ok, {"C01"}) == 1
    with pytest.raises(ValueError, match="append-only"):
        record_ratings(p, ok, {"C01"})
    for bad in [(4, "yes", "none", "plausible", ""), (2, "maybe", "none", "plausible", ""), (2, "yes", "huge", "plausible", ""),
                (2, "yes", "none", "weird", "")]:
        with pytest.raises(ValueError):
            record_ratings(p, {("C01", "B"): bad}, {"C01"})
    with pytest.raises(KeyError):
        record_ratings(p, {("C99", "A"): (2, "yes", "none", "plausible", "")}, {"C01"})


def _fixture(target_ratings, control_ratings, leak="none", covers="yes"):
    """Build mapping + ratings where column A=orig, B=V1, C=V2 for every case."""
    cases = [f"T{i}" for i in range(10)] + [f"K{i}" for i in range(6)]
    mapping = pd.DataFrame({"case_id": cases, "review_id": cases, "image_id": "x", "column_A": "orig", "column_B": "V1", "column_C": "V2"})
    rows = []
    for c in cases:
        o, v1, v2 = (target_ratings if c.startswith("T") else control_ratings)[c]
        for col, r in zip("ABC", (o, v1, v2)):
            rows.append({"case_id": c, "column": col, "rating": str(r), "covers_intended_egg": covers, "background_leakage": leak,
                         "occlusion": "plausible", "note": ""})
    return mapping, pd.DataFrame(rows), {f"T{i}" for i in range(10)}, {f"K{i}" for i in range(6)}


def test_unblinding_classification_and_criteria_known_outcome():
    # V1 improves 7 targets (2 -> 3), V2 improves 5; controls unchanged for V1, one control degraded for V2
    tr = {f"T{i}": (2, 3 if i < 7 else 2, 3 if i < 5 else 2) for i in range(10)}
    kr = {f"K{i}": (3, 3, 2 if i == 0 else 3) for i in range(6)}
    mapping, ratings, targets, controls = _fixture(tr, kr)
    res = resolve(ratings, mapping)
    cl = classify(res)
    ev = evaluate_criteria(res, cl, targets, controls).set_index("variant")
    assert ev.loc["V1", "targets_improved"] == 7 and ev.loc["V1", "ACCEPTED"]
    assert ev.loc["V2", "targets_improved"] == 5 and ev.loc["V2", "controls_degraded"] == 1 and not ev.loc["V2", "ACCEPTED"]
    assert not ev.loc["V2", "A1_targets"] and not ev.loc["V2", "A2_controls"]


def test_leakage_and_coverage_criteria_can_reject():
    tr = {f"T{i}": (1, 3, 3) for i in range(10)}
    kr = {f"K{i}": (3, 3, 3) for i in range(6)}
    mapping, ratings, targets, controls = _fixture(tr, kr, leak="obvious")
    res = resolve(ratings, mapping)
    ev = evaluate_criteria(res, classify(res), targets, controls).set_index("variant")
    assert ev.loc["V1", "A1_targets"] and not ev.loc["V1", "A3_leakage"] and not ev.loc["V1", "ACCEPTED"]
    mapping, ratings, targets, controls = _fixture(tr, kr, covers="no")
    res = resolve(ratings, mapping)
    assert not evaluate_criteria(res, classify(res), targets, controls).set_index("variant").loc["V1", "A4_coverage"]


def test_resolve_requires_all_columns_rated():
    mapping, ratings, _, _ = _fixture({f"T{i}": (2, 2, 2) for i in range(10)}, {f"K{i}": (3, 3, 3) for i in range(6)})
    with pytest.raises(ValueError, match="unrated"):
        resolve(ratings.iloc[:-1], mapping)


def test_case_ids_are_stable_in_case_order():
    cases = pd.DataFrame({"review_id": ["D9-c1", "D3-c2"]})
    assert case_ids(cases) == {"D9-c1": "C01", "D3-c2": "C02"}
