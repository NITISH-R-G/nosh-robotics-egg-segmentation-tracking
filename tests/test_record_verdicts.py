import pandas as pd
import pytest

from src.experiments.record_verdicts import record_verdicts


@pytest.fixture
def vfile(tmp_path):
    p = tmp_path / "v.csv"
    pd.DataFrame({"review_id": ["D1-c0", "D1-c1"], "image_id": ["a", "a"], "verdict": "", "uncertainty": "",
                  "note": "", "reviewer": ""}).to_csv(p, index=False)
    return p


def test_fills_blank_rows_only(vfile):
    assert record_verdicts(vfile, {"D1-c0": ("egg", "", "ok")}) == 1
    d = pd.read_csv(vfile, dtype=str, keep_default_na=False).set_index("review_id")
    assert d.loc["D1-c0", "verdict"] == "egg" and d.loc["D1-c1", "verdict"] == ""
    with pytest.raises(ValueError, match="refusing to overwrite"):
        record_verdicts(vfile, {"D1-c0": ("not_egg", "", "")})
    assert pd.read_csv(vfile, dtype=str, keep_default_na=False).set_index("review_id").loc["D1-c0", "verdict"] == "egg"


def test_rejects_invalid_input_atomically(vfile):
    with pytest.raises(ValueError):
        record_verdicts(vfile, {"D1-c1": ("maybe", "", "")})
    with pytest.raises(KeyError):
        record_verdicts(vfile, {"D9-c9": ("egg", "", "")})
    assert (pd.read_csv(vfile, dtype=str, keep_default_na=False).verdict == "").all()
