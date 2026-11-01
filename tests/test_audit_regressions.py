"""Regression tests from the pipeline root-cause audit (selection -> storage -> manifest -> review rendering).

All fixtures are synthetic and deliberately NON-SQUARE and asymmetric, so a transposition, flip, index mix-up or
silent relabelling changes the outcome.
"""
import json
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

from src.experiments import diagnose
from src.experiments.review_pool import build_items, mask_sha256, render_panel, verify_identity
from src.experiments.selector_experiments import CONFIGS, load_dev, run_config
from src.segmentation.cache_candidates import candidate_record, pack_masks, read_candidates, unpack_masks
from src.segmentation.masks import load_mask_png

H, W = 100, 150  # non-square on purpose


def _ellipse(cx, cy, a, b):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, ((cx, cy), (2 * a, 2 * b), 0), 1, -1)
    return m.astype(bool)


def _rect(x0, x1, y0, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def _cand(m, piou=0.97, stab=0.97):
    ys, xs = np.nonzero(m)
    return {"bbox_xywh": [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1], "predicted_iou": piou,
            "stability_score": stab, "point_coords": [float(xs.mean()), float(ys.mean())], "crop_box": [0, 0, W, H]}


@pytest.fixture
def world(tmp_path):
    cache = tmp_path / "cache"
    (cache / "masks").mkdir(parents=True)
    spec = {
        "img_1": [_ellipse(105, 65, 22, 14), _rect(5, 40, 60, 90)],      # egg right + a rectangle (rejected by shape)
        "img_2": [_ellipse(30, 70, 20, 12), _ellipse(110, 70, 20, 12)],  # two eggs, left and right
        "img_5": [_ellipse(75, 50, 25, 18)],                              # evaluation image
    }
    lines = []
    for img, ms in spec.items():
        cv2.imwrite(str(tmp_path / f"{img}.jpg"), np.zeros((H, W, 3), np.uint8))
        packed, shape = pack_masks(ms)
        np.savez_compressed(cache / "masks" / f"{img}.npz", masks=packed, shape=np.array(shape))
        for k, m in enumerate(ms):
            lines.append(json.dumps(candidate_record(img, k, f"masks/{img}.npz", m, _cand(m))))
    (cache / "candidates.jsonl").write_text("\n".join(lines) + "\n")
    split = tmp_path / "split.csv"
    pd.DataFrame({"image_id": ["img_1", "img_2", "img_5"], "split": ["dev", "dev", "eval"],
                  "path": ["img_1.jpg", "img_2.jpg", "img_5.jpg"], "split_version": "t"}).to_csv(split, index=False)
    counts = tmp_path / "counts.csv"
    pd.DataFrame({"image_id": ["img_1", "img_2"], "expected_egg_count": [1, 2], "difficulty_tag": ["", ""]}).to_csv(counts, index=False)
    return tmp_path, cache, split, counts, spec


def _run(world, name="S1_conservative", save=True):
    tmp, cache, split, counts, spec = world
    cands, c = load_dev(cache, split, counts)
    paths = {"img_1": "img_1.jpg", "img_2": "img_2.jpg"}
    out = tmp / "out" / name if save else None
    res, _ = run_config(CONFIGS[name], cands, c, cache, tmp, paths, out)
    return res, out


# 1. selected candidate ids map to the exact masks stored in the cache
def test_selected_ids_map_to_exact_cached_masks_and_saved_union(world):
    tmp, cache, split, counts, spec = world
    res, out = _run(world)
    for r in res.itertuples():
        stored = unpack_masks(cache / "masks" / f"{r.image_id}.npz")
        want = np.zeros((H, W), bool)
        for i in r.selected_idx:
            assert (stored[i] == spec[r.image_id][i]).all()          # same mask bytes as the fixture's candidate i
            want |= spec[r.image_id][i]
        saved = load_mask_png(out / "masks" / f"{r.image_id}.png", (H, W)).astype(bool)
        assert (saved == want).all()
    sel = {r.image_id: list(r.selected_idx) for r in res.itertuples()}
    assert sel == {"img_1": [0], "img_2": [0, 1]} or sel == {"img_1": [0], "img_2": [1, 0]}  # rectangle rejected, eggs kept


# 2. overlays are drawn in the correct (x, y) frame without resizing/transposition
def test_overlay_contour_is_in_image_xy_frame(world):
    tmp, cache, split, counts, spec = world
    items = pd.DataFrame({"review_id": ["D1-c0"], "candidate_index": [0]})
    panel = render_panel("img_1", items, cache, tmp, "img_1.jpg")
    assert panel.shape == (H, W, 3)  # no resizing
    drawn = (panel[40:, :, :] != 0).any(axis=2)  # below the label banner (rows 0-35)
    ys, xs = np.nonzero(drawn)
    m_ys, m_xs = np.nonzero(spec["img_1"][0])
    assert ys.size and abs(xs.min() - m_xs.min()) <= 3 and abs(xs.max() - m_xs.max()) <= 3
    assert abs((ys.min() + 40) - m_ys.min()) <= 3 and abs((ys.max() + 40) - m_ys.max()) <= 3  # rows were offset by the slice
    assert not drawn[:, :60].any()  # nothing drawn where a transposed/mirrored egg would be (left side)


def test_contour_check_detects_transposed_and_shifted_masks():
    sq = np.zeros((80, 80), bool)
    sq[10:30, 40:70] = True
    assert diagnose.contour_matches_mask(sq)
    assert not diagnose.contour_matches_mask(sq, drawn=sq.T)                    # transposed overlay
    assert not diagnose.contour_matches_mask(sq, drawn=np.roll(sq, 15, axis=1))  # shifted overlay
    assert not diagnose.contour_matches_mask(sq, drawn=sq[::-1])                 # vertically flipped overlay
    border = np.zeros((80, 80), bool)
    border[0:20, 0:30] = True
    assert diagnose.contour_matches_mask(border)  # border-touching masks must pass (false alarm fixed in this audit)


def test_prompt_point_distance_separates_correct_from_transposed(world):
    tmp, cache, split, counts, spec = world
    m = spec["img_2"][1]
    ys, xs = np.nonzero(m)
    assert diagnose.prompt_point_distance(m, xs.mean(), ys.mean()) == 0.0
    assert diagnose.prompt_point_distance(m, 30.0, 70.0) > 20  # the OTHER egg's centre is far from this mask


# 3. candidate ids stay consistent through selection -> per-image csv -> review items -> provenance -> rendering
def test_candidate_ids_consistent_end_to_end(world):
    tmp, cache, split, counts, spec = world
    for name in ("S0_pilot_filter", "S1_conservative"):
        res, _ = _run(world, name, save=False)
        res.assign(selected_idx=res.selected_idx.map(json.dumps), config_name=name, experiment_id=f"{name}@t",
                   code_version="t").to_csv(tmp / f"selector_{name}_2026_per_image.csv", index=False)
    items, prov = build_items(str(tmp / "selector_*_2026_per_image.csv"), cache, split)
    cands = read_candidates(cache, {"img_1", "img_2"})
    for r in items.itertuples():
        stored = unpack_masks(cache / "masks" / f"{r.image_id}.npz")
        assert mask_sha256(stored[r.candidate_index]) == r.mask_sha256 == mask_sha256(spec[r.image_id][r.candidate_index])
        assert r.review_id == f"D{r.image_id.split('_')[1]}-c{r.candidate_index}"
        assert int(cands[(cands.image_id == r.image_id) & (cands["index"] == r.candidate_index)].iloc[0].area) == r.area
    for p in prov.itertuples():
        img, idx = p.candidate_id.split("#")
        assert mask_sha256(spec[img][int(idx)]) == p.mask_sha256
    # every rendered item draws on its own mask: panel pixels appear near each item's bbox
    for img in ("img_1", "img_2"):
        panel = render_panel(img, items[items.image_id == img], cache, tmp, f"{img}.jpg")
        for r in items[items.image_id == img].itertuples():
            ys, xs = np.nonzero(spec[img][r.candidate_index])
            region = panel[max(ys.min() - 2, 40):ys.max() + 3, xs.min() - 2:xs.max() + 3]
            assert region.any()


# 4. a deliberately wrong candidate cannot be silently relabelled as another candidate
def test_swapped_masks_in_cache_are_detected(world):
    tmp, cache, split, counts, spec = world
    z = np.load(cache / "masks" / "img_2.npz")
    swapped = z["masks"][::-1].copy()  # index 0 now holds the right-hand egg, but the manifest still says "left egg"
    np.savez_compressed(cache / "masks" / "img_2.npz", masks=swapped, shape=z["shape"])
    cands = read_candidates(cache, {"img_2"})
    masks = unpack_masks(cache / "masks" / "img_2.npz")
    with pytest.raises(ValueError, match="disagrees with its manifest record"):
        verify_identity(cands, masks, "img_2", 0)
    run = tmp / "selector_X_2026_per_image.csv"
    pd.DataFrame({"image_id": ["img_2"], "selected_idx": ["[0]"], "config_name": "X", "experiment_id": "X@t",
                  "code_version": "t"}).to_csv(run, index=False)
    with pytest.raises(ValueError, match="refusing to treat it as this candidate"):
        build_items(str(run), cache, split)


def test_out_of_range_and_unknown_candidates_raise(world):
    tmp, cache, split, counts, spec = world
    cands = read_candidates(cache, {"img_1"})
    masks = unpack_masks(cache / "masks" / "img_1.npz")
    with pytest.raises(IndexError):
        verify_identity(cands, masks, "img_1", 7)
    with pytest.raises(KeyError):
        verify_identity(cands.iloc[:1], masks, "img_1", 1)  # manifest row for #1 missing


def test_review_sheet_context_marks_reviewed_items_gray(world):
    tmp, cache, split, counts, spec = world
    new = pd.DataFrame({"review_id": ["D2-c0"], "candidate_index": [0]})
    ctx = pd.DataFrame({"review_id": ["D2-c1"], "candidate_index": [1], "verdict": ["egg"]})
    with_ctx = render_panel("img_2", new, cache, tmp, "img_2.jpg", context=ctx)
    without = render_panel("img_2", new, cache, tmp, "img_2.jpg")
    right = slice(48, 95), slice(88, 135)  # region around the right-hand egg (candidate 1)
    assert with_ctx[right].any() and not without[right].any()  # the already-reviewed egg is no longer hidden


# 5. evaluation data is rejected by the diagnostic and experiment entry points
def test_diagnose_rejects_eval_before_loading_anything(world, monkeypatch, tmp_path):
    tmp, cache, split, counts, spec = world
    with pytest.raises(ValueError, match="non-development"):
        diagnose.assert_dev(["img_1", "img_5"], split)
    with pytest.raises(ValueError, match="unknown image ids"):
        diagnose.assert_dev(["nope"], split)
    monkeypatch.setattr(sys, "argv", ["x", "--images", "5", "--cache", str(tmp_path / "does_not_exist"), "--split", str(split),
                                      "--s1-run", "x.csv"])
    # image id "not_damaged_5" is unknown here; use the real naming by renaming the fixture ids
    d = pd.read_csv(split)
    d["image_id"] = d.image_id.str.replace("img_", "not_damaged_")
    d.to_csv(split, index=False)
    with pytest.raises(ValueError, match="non-development"):
        diagnose.main()  # raises before the (missing) cache directory is ever touched


def test_candidate_reader_and_experiment_loader_never_parse_eval_lines(world):
    tmp, cache, split, counts, spec = world
    lines = (cache / "candidates.jsonl").read_text().splitlines()
    lines.insert(0, '{"image_id": "img_5", THIS IS NOT JSON')
    (cache / "candidates.jsonl").write_text("\n".join(lines) + "\n")
    assert set(read_candidates(cache, {"img_1", "img_2"}).image_id) == {"img_1", "img_2"}
    cands, _ = load_dev(cache, split, counts)
    assert "img_5" not in set(cands.image_id)


def test_review_builder_rejects_eval_runs(world):
    tmp, cache, split, counts, spec = world
    run = tmp / "selector_E_2026_per_image.csv"
    pd.DataFrame({"image_id": ["img_5"], "selected_idx": ["[0]"], "config_name": "E", "experiment_id": "E@t",
                  "code_version": "t"}).to_csv(run, index=False)
    with pytest.raises(ValueError, match="non-dev"):
        build_items(str(run), cache, split)
