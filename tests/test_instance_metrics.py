import pandas as pd
import pytest

from src.experiments.instance_metrics import instance_metrics


def _data():
    prov = pd.DataFrame({
        "review_id": ["D1-c0", "D1-c1", "D2-c0", "D1-c0", "D2-c0"],
        "image_id": ["i1", "i1", "i2", "i1", "i2"],
        "candidate_id": ["i1#0", "i1#1", "i2#0", "i1#0", "i2#0"],
        "config_name": ["A", "A", "A", "B", "B"]})
    ver = pd.DataFrame({"review_id": ["D1-c0", "D1-c1", "D2-c0"], "verdict": ["egg", "not_egg", "egg_bad_boundary"]})
    counts = pd.DataFrame({"image_id": ["i1", "i2", "i3"], "expected_egg_count": [1, 2, 1]})
    return prov, ver, counts


def test_known_precision_recall():
    agg, per = instance_metrics(*_data())
    a = agg.set_index("config_name").loc["A"]
    assert (a.selected, a.tp_strict, a.tp_lenient, a.false_positives, a.missed_eggs, a.labelled_eggs) == (3, 1, 2, 1, 2, 4)
    assert a.precision_lenient == pytest.approx(2 / 3, abs=1e-3) and a.recall_lenient == 0.5 and a.recall_strict == 0.25
    b = agg.set_index("config_name").loc["B"]
    assert (b.false_positives, b.tp_lenient, b.missed_eggs) == (0, 2, 2) and b.precision_lenient == 1.0


def test_blank_verdict_is_an_error():
    prov, ver, counts = _data()
    ver.loc[0, "verdict"] = ""
    with pytest.raises(ValueError, match="no verdict"):
        instance_metrics(prov, ver, counts)


def test_tp_exceeding_label_is_flagged_not_hidden():
    prov, ver, counts = _data()
    counts.loc[0, "expected_egg_count"] = 0
    agg, per = instance_metrics(prov, ver, counts)
    assert agg.set_index("config_name").loc["A"].tp_exceeds_label_images == 1
