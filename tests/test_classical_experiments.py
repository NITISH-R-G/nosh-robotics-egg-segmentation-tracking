import numpy as np
import pandas as pd
import pytest

from src.experiments.classical_experiments import (EXPERIMENTS, RULE, aggregate, decide, key_scores, run_sequence, subsets)
from src.segmentation.classical import ClassicalConfig


def _df(rows):
    """rows: (image_id, scene, rev_tp, rev_fp, rev_fn, n_pred, expected, rev_ref_empty, rev_pred_empty, rev_iou)"""
    out = []
    for i, sc, tp, fp, fn, n, e, rre, rpe, iou in rows:
        r = {"image_id": i, "scene": sc, "tag": "", "expected": e, "n_pred": n, "runtime_s": 0.01, "pred_fg_px": 10}
        for nm in ("raw", "rev"):
            r.update({f"{nm}_ref_empty": rre, f"{nm}_pred_empty": rpe, f"{nm}_iou": iou, f"{nm}_dice": iou, f"{nm}_tp_px": 50, f"{nm}_fp_px": 10,
                      f"{nm}_fn_px": 20, f"{nm}_inst_tp": tp, f"{nm}_inst_fp": fp, f"{nm}_inst_fn": fn})
        out.append(r)
    return pd.DataFrame(out)


def test_experiment_list_is_closed_and_matches_the_amendment():
    assert [e[0] for e in EXPERIMENTS] == [f"E{i:02d}" for i in range(14)]
    assert EXPERIMENTS[0][1] == {} and [e[3] for e in EXPERIMENTS[1:4]] == ["colour"] * 3 and all(e[3] == "single" for e in EXPERIMENTS[4:])
    assert RULE == {"min_gain_f1": 0.02, "max_loss_iou": 0.01, "max_loss_exact": 0.03}
    assert ClassicalConfig().colour == "L" and ClassicalConfig().open_k == 5 and ClassicalConfig().min_area_frac == 0.005  # E00 as written


def test_aggregate_macro_excludes_empty_references_and_reports_them_separately():
    df = _df([("a", "s1", 1, 0, 0, 1, 1, False, False, 0.9), ("b", "s1", 0, 2, 0, 2, 0, True, False, float("nan")),
              ("c", "s2", 0, 0, 1, 0, 1, False, True, 0.0)])
    a = aggregate(df)["rev"]
    assert a["macro_iou"] == pytest.approx(0.45)  # (0.9 + 0.0) / 2; image b (empty reference) excluded
    assert a["n_images_ref_empty"] == 1 and a["images_ref_empty_with_prediction"] == 1 and a["images_empty_prediction_nonempty_ref"] == 1
    assert a["fp_px_in_ref_empty_images"] == 10
    assert (a["inst_tp"], a["inst_fp"], a["inst_fn"]) == (1, 2, 1) and a["inst_f1"] == pytest.approx(2 * 1 / (2 * 1 + 2 + 1))
    assert aggregate(df, {"a"})["n_images"] == 1


def test_counts_are_not_the_only_signal_false_positive_and_miss_both_visible():
    df = _df([("a", "s", 1, 1, 1, 2, 2, False, False, 0.5)])  # count matches (2 vs 2) but one egg wrong and one missed
    a = aggregate(df)
    assert a["counts"]["exact_rate"] == 1.0 and a["rev"]["inst_fp"] == 1 and a["rev"]["inst_fn"] == 1


def test_decision_rule_thresholds_exact():
    base = {"f1_rev": 0.50, "iou_rev": 0.60, "exact": 0.50}
    assert decide(base, {"f1_rev": 0.52, "iou_rev": 0.60, "exact": 0.50})[0]                 # gain exactly 0.02 -> kept
    assert not decide(base, {"f1_rev": 0.519, "iou_rev": 0.60, "exact": 0.50})[0]             # gain too small
    assert not decide(base, {"f1_rev": 0.60, "iou_rev": 0.58, "exact": 0.50})[0]              # IoU loss > 0.01
    assert decide(base, {"f1_rev": 0.60, "iou_rev": 0.59, "exact": 0.47})[0]                  # IoU loss 0.01, exact loss 0.03 allowed
    assert not decide(base, {"f1_rev": 0.60, "iou_rev": 0.60, "exact": 0.46})[0]              # exact loss > 0.03


def test_greedy_sequence_uses_decision_rule_and_chooses_best_colour_only():
    # fake evaluator: score depends on the config so we know exactly what should be kept
    def make(f1, iou=0.6):
        # 10 images each with tp = round(100*f1) and fp = fn = 100 - tp  ->  micro F1 = 2tp / (2tp + fp + fn) = tp / 100 = f1 exactly
        tp = round(f1 * 100)
        return _df([(f"i{k}", "s", tp, 100 - tp, 100 - tp, 1, 1, False, False, iou) for k in range(10)])

    def ev(cfg):
        f1 = 0.40 + (0.10 if cfg.colour == "W" else 0.06 if cfg.colour == "V" else 0.0) + (0.05 if cfg.blur_sigma > 0 else 0.0) \
            + (0.03 if cfg.min_solidity > 0 else 0.0) + (0.01 if cfg.split_watershed else 0.0)
        return make(f1)

    cfg, steps = run_sequence(ev, {f"i{k}" for k in range(10)})
    dec = {s["id"]: s["decision"] for s in steps}
    assert cfg.colour == "W" and dec["E02"] == "kept" and dec["E01"] == "discarded" and dec["E03"] == "discarded"  # best of the kept colour trials
    assert dec["E04"] == "kept" and cfg.blur_sigma == 2.0
    assert dec["E08"] == "kept" and cfg.min_solidity == 0.85
    assert dec["E12"] == "discarded" and not cfg.split_watershed  # +0.01 < 0.02 gain
    assert dec["E00"] == "baseline" and len(steps) == 14


def _scene_inputs():
    ids = [f"i{k}" for k in range(9)]
    return {"ids": ids, "scene": {i: "ABC"[k // 3] for k, i in enumerate(ids)}}


def _fake_eval(poison_scene=None):
    """Per-config evaluator; optionally replaces ALL results of one scene by garbage (a leak detector)."""
    inputs = _scene_inputs()

    def ev(cfg):
        f1 = 0.40 + (0.10 if cfg.colour == "W" else 0.0) + (0.05 if cfg.blur_sigma > 0 else 0.0)
        rows = []
        for i in inputs["ids"]:
            sc = inputs["scene"][i]
            tp = 5 if sc == poison_scene else round(f1 * 100)  # poisoned scene: constant, uninformative
            rows.append((i, sc, tp, 100 - tp, 100 - tp, 1, 1, False, False, 0.6 if sc != poison_scene else 0.0))
        return _df(rows)
    return ev


def test_fold_selection_ignores_the_held_out_scene_completely():
    from src.experiments.classical_experiments import loso

    inputs = _scene_inputs()
    for held in "ABC":
        train = {i for i in inputs["ids"] if inputs["scene"][i] != held}
        clean_cfg, clean_steps = run_sequence(_fake_eval(None), train)
        poisoned_cfg, poisoned_steps = run_sequence(_fake_eval(held), train)  # the held-out scene's results are destroyed
        assert clean_cfg == poisoned_cfg
        assert [s["decision"] for s in clean_steps] == [s["decision"] for s in poisoned_steps]
        assert all(s["scores"] == p["scores"] for s, p in zip(clean_steps, poisoned_steps))
    out = loso(_fake_eval(None), inputs)
    assert sorted(out.held_out_scene.unique()) == ["A", "B", "C"] and set(out.reference) == {"raw", "rev"} and (out.n_images == 3).all()
    # the held-out scene is evaluated only AFTER selection: its own results never enter that fold's scores
    train_a = {i for i in inputs["ids"] if inputs["scene"][i] != "A"}
    assert set(train_a).isdisjoint({"i0", "i1", "i2"})


def test_subsets_follow_the_amendment(tmp_path):
    man = pd.DataFrame({"image_id": ["i1", "i2", "i3", "i4"], "action": ["add_to_reviewed_reference", "add_to_reviewed_reference", "exclude", "exclude"],
                        "verification_status": ["first_pass_only", "confirmed", "confirmed", "uncertain"]})
    p = tmp_path / "m.csv"
    man.to_csv(p, index=False)
    s = subsets([f"i{k}" for k in range(1, 8)], p)
    assert len(s["S_all"]) == 7 and s["S_no_first_pass"] == s["S_all"] - {"i1"} and s["S_clean"] == {"i5", "i6", "i7"}
