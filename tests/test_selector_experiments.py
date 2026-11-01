import json

import cv2
import numpy as np
import pandas as pd
import pytest

from src.experiments.selector_experiments import CONFIGS, load_dev, run_config, summarize
from src.segmentation.cache_candidates import candidate_record, pack_masks

H = W = 100


def ell(cx, cy, a, b):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, ((cx, cy), (2 * a, 2 * b), 0), 1, -1)
    return m.astype(bool)


def _cand(m, piou=0.97, stab=0.97):
    ys, xs = np.nonzero(m)
    return {"bbox_xywh": [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1],
            "predicted_iou": piou, "stability_score": stab, "point_coords": [1.0, 1.0], "crop_box": [0, 0, W, H]}


@pytest.fixture
def world(tmp_path):
    cache = tmp_path / "cache"
    (cache / "masks").mkdir(parents=True)
    spec = {  # image -> list of (mask, piou, stab)
        "img_1": [(ell(50, 50, 25, 18), 0.97, 0.97)],                                  # clean single egg
        "img_2": [(ell(30, 30, 20, 14), 0.97, 0.97), (ell(70, 70, 20, 14), 0.97, 0.97)],  # two eggs
        "img_3": [(ell(50, 50, 25, 18), 0.85, 0.87)],                                  # weak candidate: only relaxed rules
        "img_4": [],                                                                    # zero candidates
        "img_5": [(ell(50, 50, 25, 18), 0.97, 0.97)],                                  # eval image, must never load
    }
    lines = []
    for img, items in spec.items():
        if items:
            packed, shape = pack_masks([m for m, _, _ in items])
        else:
            packed, shape = np.zeros((0, 0), np.uint8), (H, W)
        np.savez_compressed(cache / "masks" / f"{img}.npz", masks=packed, shape=np.array(shape))
        for k, (m, p, s) in enumerate(items):
            lines.append(json.dumps(candidate_record(img, k, f"masks/{img}.npz", m, _cand(m, p, s))))
    lines.insert(0, '{"image_id": "img_5", THIS IS NOT JSON BUT MUST NEVER BE PARSED')
    (cache / "candidates.jsonl").write_text("\n".join(lines) + "\n")
    split = tmp_path / "split.csv"
    pd.DataFrame({"image_id": [f"img_{i}" for i in range(1, 6)], "split": ["dev"] * 4 + ["eval"],
                  "path": [f"{i}.jpg" for i in range(1, 6)], "split_version": "vtest"}).to_csv(split, index=False)
    counts = tmp_path / "counts.csv"
    pd.DataFrame({"image_id": ["img_1", "img_2", "img_3", "img_4"], "expected_egg_count": [1, 2, 1, 1],
                  "difficulty_tag": ["", "", "submerged", ""]}).to_csv(counts, index=False)
    return cache, split, counts


def test_eval_candidate_lines_are_never_parsed_or_returned(world):
    cache, split, counts = world
    cands, c = load_dev(cache, split, counts)  # would raise JSONDecodeError if the eval line were parsed
    assert "img_5" not in set(cands.image_id) and set(cands.image_id) == {"img_1", "img_2", "img_3"}


def test_non_dev_labels_rejected(world, tmp_path):
    cache, split, counts = world
    bad = pd.read_csv(counts)
    bad.loc[0, "image_id"] = "img_5"
    bad.to_csv(counts, index=False)
    with pytest.raises(ValueError, match="non-dev"):
        load_dev(cache, split, counts)


def _run_all(world):
    cache, split, counts = world
    cands, c = load_dev(cache, split, counts)
    paths = {f"img_{i}": f"{i}.jpg" for i in range(1, 6)}
    return {n: run_config(cfg, cands, c, cache, cache, paths, None)[0] for n, cfg in CONFIGS.items()}


def test_all_configs_see_identical_images_in_identical_order(world):
    res = _run_all(world)
    ids = {n: list(r.image_id) for n, r in res.items()}
    assert len({tuple(v) for v in ids.values()}) == 1 and ids["S0_pilot_filter"] == ["img_1", "img_2", "img_3", "img_4"]
    exp = {n: list(r.expected) for n, r in res.items()}
    assert len({tuple(v) for v in exp.values()}) == 1


def test_s1_and_s3_select_exactly_the_same_candidates(world):
    res = _run_all(world)
    s1, s3 = res["S1_conservative"], res["S3_S1_plus_review_flag"]
    assert [list(v) for v in s1.selected_idx] == [list(v) for v in s3.selected_idx]
    # the only difference is the status of empty selections
    diff = s1.status != s3.status
    assert (s3.loc[diff, "selected"] == 0).all() and (s3.loc[diff, "status"] == "review_no_candidate").all()
    assert (s1.loc[diff, "status"] == "none_selected").all()
    assert diff.any()  # fixture contains empty selections, so the flag is exercised


def test_s4_differs_from_s1_only_where_s1_selects_nothing(world):
    res = _run_all(world)
    s1, s4 = res["S1_conservative"], res["S4_S1_plus_relaxed_fallback"]
    same = s1.selected > 0
    assert [list(v) for v in s1.selected_idx[same]] == [list(v) for v in s4.selected_idx[same]]
    assert (s4.loc[~same, "selected"] <= 1).all()
    assert (s4.loc[~same & (s4.selected == 1), "status"] == "fallback_top1").all()


def test_s2_equals_s1_when_no_candidate_touches_border(world):
    res = _run_all(world)
    assert [list(v) for v in res["S1_conservative"].selected_idx] == [list(v) for v in res["S2_conservative_border_rule"].selected_idx]


def test_summarize_reports_median_and_tag_breakdown(world):
    r = _run_all(world)["S1_conservative"]
    m = summarize(r)
    assert set(m) == {"all_dev", "excluding_submerged", "by_tag"} and "submerged" in m["by_tag"]
    assert m["all_dev"]["n_images"] == 4 and m["excluding_submerged"]["n_images"] == 3
    assert "median_abs_count_err" in m["all_dev"]


# ---------------- gate / log-integrity tests ----------------
import hashlib
import sys

import src.experiments.selector_experiments as se


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_audit(cache, ok=True, **over):
    (cache / "run.json").write_text('{"format_version": 1}')
    hashes = {n: _sha(cache / n) for n in ("candidates.jsonl", "run.json")}
    (cache / "audit.json").write_text(json.dumps({"ok": ok, "n_errors": 0 if ok else 1, "stats": {"hashes": hashes, **over}}))


def test_audit_gate_requires_passing_and_current_audit(world):
    cache, split, counts = world
    with pytest.raises(RuntimeError, match="no audit artifact"):
        se.require_passing_audit(cache)
    _write_audit(cache, ok=False)
    with pytest.raises(RuntimeError, match="did not pass"):
        se.require_passing_audit(cache)
    _write_audit(cache, ok=True)
    assert se.require_passing_audit(cache)["ok"]
    with open(cache / "candidates.jsonl", "a") as f:  # cache modified after the audit -> stale
        f.write("\n")
    with pytest.raises(RuntimeError, match="stale audit"):
        se.require_passing_audit(cache)


def test_unknown_split_values_are_never_treated_as_dev(world):
    cache, split, counts = world
    d = pd.read_csv(split)
    d.loc[0, "split"] = "Dev"  # case variant
    d.to_csv(split, index=False)
    with pytest.raises(ValueError, match="unknown split values"):
        load_dev(cache, split, counts)
    d.loc[0, "split"] = None
    d.to_csv(split, index=False)
    with pytest.raises(ValueError, match="unknown split values"):
        load_dev(cache, split, counts)


def test_results_do_not_depend_on_input_row_order(world, tmp_path):
    cache, split, counts = world
    a = _run_all(world)
    lines = (cache / "candidates.jsonl").read_text().splitlines()
    (cache / "candidates.jsonl").write_text("\n".join([lines[0]] + lines[1:][::-1]) + "\n")
    c = pd.read_csv(counts).iloc[::-1]
    c.to_csv(counts, index=False)
    b = _run_all(world)
    for n in a:
        assert list(a[n].image_id) == list(b[n].image_id)
        assert [list(v) for v in a[n].selected_idx] == [list(v) for v in b[n].selected_idx]


def _run_main(world, monkeypatch, tmp_path, extra=()):
    cache, split, counts = world
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(se, "LOG", tmp_path / "experiments" / "log.csv")
    monkeypatch.setattr(sys, "argv", ["x", "--cache", str(cache), "--split", str(split), "--counts", str(counts),
                                       "--data", str(cache), *extra])
    se.main()


def test_failed_preflight_is_logged_and_reraised(world, monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="no audit artifact"):
        _run_main(world, monkeypatch, tmp_path)
    log = pd.read_csv(tmp_path / "experiments" / "log.csv")
    assert log.status.iloc[0].startswith("failed_preflight") and log.experiment_id.iloc[0].startswith("preflight@")


def test_logged_command_and_hashes_match_what_ran(world, monkeypatch, tmp_path):
    cache, split, counts = world
    _write_audit(cache)
    _run_main(world, monkeypatch, tmp_path, extra=("--configs", "S0_pilot_filter", "S1_conservative"))
    log = pd.read_csv(tmp_path / "experiments" / "log.csv")
    assert list(log.status) == ["completed", "completed"] and len(log) == 2
    assert list(log.experiment_id.str.split("@").str[0]) == ["S0_pilot_filter", "S1_conservative"]
    resolved = json.loads(log.command.iloc[0].split("selector_experiments ", 1)[1])
    assert resolved["configs"] == ["S0_pilot_filter", "S1_conservative"] and resolved["split"] == str(split)
    assert log.split_sha256.iloc[0] == _sha(split)[:16] and log.counts_sha256.iloc[0] == _sha(counts)[:16]
    assert log.candidates_sha256.iloc[0] == _sha(cache / "candidates.jsonl")[:16]
    assert log.cache_run_sha256.iloc[0] == _sha(cache / "run.json")[:16]
    assert (log.n_dev_images == 4).all() and log.split_version.iloc[0] == "vtest"
    assert json.loads(log.config.iloc[1])["min_ellipse_iou"] == 0.93  # actual config, not a label
    assert len(list((tmp_path / "experiments").glob("selector_*_per_image.csv"))) == 2


def test_failed_config_run_is_logged(world, monkeypatch, tmp_path):
    _write_audit(world[0])
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(se, "run_config", boom)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        _run_main(world, monkeypatch, tmp_path, extra=("--configs", "S0_pilot_filter"))
    log = pd.read_csv(tmp_path / "experiments" / "log.csv")
    assert log.status.iloc[0] == "failed: RuntimeError: synthetic failure"
