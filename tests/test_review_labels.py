from pathlib import Path

import pandas as pd
import pytest

from src.data.review_labels import validate_counts

ROOT = Path(__file__).parents[1] / "data"
AI = "AI-assisted (test)"


def _split(n_dev=4, n_eval=2):
    ids = [f"img_{i}" for i in range(n_dev + n_eval)]
    return pd.DataFrame({"image_id": ids, "split": ["dev"] * n_dev + ["eval"] * n_eval})


def _counts(n=4, **over):
    d = pd.DataFrame({"image_id": [f"img_{i}" for i in range(n)], "expected_egg_count": [1] * n,
                      "pass1_count": [1] * n, "annotation_process": [AI] * n})
    for k, v in over.items():
        d[k] = v
    return d


NO_CORR = pd.DataFrame(columns=["image_id", "old_count", "new_count", "reason"])


def test_valid_fixture_passes():
    assert validate_counts(_counts(), _split(), NO_CORR) == []


def test_duplicate_missing_and_eval_ids_detected():
    c = _counts(); c.loc[1, "image_id"] = "img_0"
    e = validate_counts(c, _split(), NO_CORR)
    assert any("duplicate" in x for x in e) and any("without a label" in x for x in e)
    c = pd.concat([_counts(), pd.DataFrame({"image_id": ["img_4"], "expected_egg_count": [1], "pass1_count": [1],
                                            "annotation_process": [AI]})], ignore_index=True)
    assert any("eval images present" in x for x in validate_counts(c, _split(), NO_CORR))


def test_unknown_id_and_wrong_total_detected():
    c = _counts(); c.loc[0, "image_id"] = "ghost"
    e = validate_counts(c, _split(), NO_CORR)
    assert any("not in split manifest" in x for x in e)


@pytest.mark.parametrize("bad", [-1, 1.5, "two", None])
def test_invalid_counts_detected(bad):
    c = _counts(); c["expected_egg_count"] = c["expected_egg_count"].astype(object); c.loc[2, "expected_egg_count"] = bad
    assert any("non-negative integers" in x for x in validate_counts(c, _split(), NO_CORR))


def test_correction_requires_record_and_reason():
    c = _counts(); c.loc[0, "expected_egg_count"] = 2
    assert any("without correction record" in x for x in validate_counts(c, _split(), NO_CORR))
    corr = pd.DataFrame({"image_id": ["img_0"], "old_count": [1], "new_count": [2], "reason": [""]})
    assert any("no reason" in x for x in validate_counts(c, _split(), corr))
    corr["reason"] = ["second look: two eggs"]
    assert validate_counts(c, _split(), corr) == []
    corr["new_count"] = [3]
    assert any("does not match" in x for x in validate_counts(c, _split(), corr))


def test_missing_columns_and_ai_disclosure():
    assert any("missing columns" in x for x in validate_counts(_counts().drop(columns="pass1_count"), _split(), NO_CORR))
    c = _counts(annotation_process="manual")
    assert any("AI-assisted" in x for x in validate_counts(c, _split(), NO_CORR))


# ---- real repository data (relative paths only) ----
@pytest.fixture(scope="module")
def real():
    return (pd.read_csv(ROOT / "review" / "dev_egg_counts.csv"), pd.read_csv(ROOT / "split_manifest.csv"),
            pd.read_csv(ROOT / "review" / "dev_egg_counts_corrections.csv"))


def test_real_manifest_is_valid(real):
    c, s, corr = real
    assert validate_counts(c, s, corr) == []
    assert len(c) == 88 and (s.split == "dev").sum() == 88 and (s.split == "eval").sum() == 22
    assert set(s.split_version) == {"v2_pilot_exposure_controlled"}


def test_real_all_88_rows_present_and_image_117_correction(real):
    c, s, corr = real
    assert set(c.image_id) == set(s.loc[s.split == "dev", "image_id"])
    r = c.set_index("image_id").loc["not_damaged_117"]
    assert r.expected_egg_count == 1 and r.pass1_count == 2
    cr = corr.set_index("image_id").loc["not_damaged_117"]
    assert (cr.old_count, cr.new_count) == (2, 1) and "depressions" in cr.reason
    assert "empty tray depressions" in r.second_pass


def test_real_duplicate_groups_consistent(real):
    c = real[0].set_index("image_id").expected_egg_count
    assert c["not_damaged_78"] == c["not_damaged_27"] == 1 and c["not_damaged_156"] == c["not_damaged_163"] == 1
