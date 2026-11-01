import pandas as pd
import pytest

from src.experiments.verify_verdicts import effective_verdicts, record_verification, verification_set

ITEMS = pd.DataFrame({"review_id": ["D1-c0", "D1-c5", "D2-c1", "D3-c2"], "image_id": ["a", "a", "b", "c"],
                      "canonical_candidate_id": ["a#0", "a#5", "b#1", "c#2"], "candidate_index": [0, 5, 1, 2],
                      "mask_sha256": ["h0", "h5", "h1", "h2"]})
VERD = pd.DataFrame({"review_id": ITEMS.review_id, "image_id": ITEMS.image_id,
                     "verdict": ["egg", "not_egg", "egg_bad_boundary", "not_egg"]})
PROV = pd.DataFrame({"review_id": ["D1-c0", "D1-c5", "D2-c1", "D3-c2"], "image_id": ["a", "a", "b", "c"],
                     "config_name": ["S1_conservative", "E1_x", "S0_pilot", "E2_x"]})


def test_verification_set_is_round2_only_plus_bad_boundary_and_skips_checked():
    got = set(verification_set(ITEMS, VERD, PROV, already_checked={"D3-c2"}).review_id)
    assert got == {"D1-c5", "D2-c1"}  # D1-c5 round-2 only; D2-c1 is a bad-boundary item; D3-c2 was already checked


def test_record_is_append_only_validated_and_traceable(tmp_path):
    p = tmp_path / "ver.csv"
    n = record_verification(p, ITEMS, VERD, {"D1-c5": ("confirmed", "not_egg", "clear_not_egg", "cloth patch"),
                                            "D2-c1": ("corrected", "egg", "partial_view", "covers the visible egg")})
    assert n == 2
    d = pd.read_csv(p, dtype=str, keep_default_na=False).set_index("review_id")
    assert d.loc["D1-c5", "candidate_id"] == "a#5" and d.loc["D1-c5", "mask_sha256"] == "h5"
    assert "not human-verified" in d.loc["D2-c1", "reviewer"] and d.loc["D2-c1", "original_verdict"] == "egg_bad_boundary"
    with pytest.raises(ValueError, match="append-only"):
        record_verification(p, ITEMS, VERD, {"D1-c5": ("confirmed", "not_egg", "clear_not_egg", "again")})


@pytest.mark.parametrize("entry", [("confirmed", "egg", "", "x"), ("corrected", "not_egg", "", "x"),
                                   ("maybe", "not_egg", "", "x"), ("confirmed", "not_egg", "", " "),
                                   ("confirmed", "not_egg", "bogus", "x"), ("confirmed", "unsure", "", "x")])
def test_invalid_entries_rejected(tmp_path, entry):
    with pytest.raises(ValueError):
        record_verification(tmp_path / "v.csv", ITEMS, VERD, {"D1-c5": entry})
    assert not (tmp_path / "v.csv").exists()


def test_unknown_review_id_rejected(tmp_path):
    with pytest.raises(KeyError):
        record_verification(tmp_path / "v.csv", ITEMS, VERD, {"D9-c9": ("confirmed", "not_egg", "", "x")})


def test_effective_verdicts_apply_corrections_without_touching_originals(tmp_path):
    p = tmp_path / "ver.csv"
    record_verification(p, ITEMS, VERD, {"D2-c1": ("corrected", "egg", "partial_view", "covers the visible egg"),
                                         "D1-c5": ("uncertain", "not_egg", "ambiguous_object", "cannot tell")})
    eff = effective_verdicts(VERD, pd.read_csv(p, dtype=str, keep_default_na=False)).set_index("review_id")
    assert eff.loc["D2-c1", "verdict"] == "egg" and eff.loc["D2-c1", "original_verdict"] == "egg_bad_boundary"
    assert eff.loc["D1-c5", "verification_status"] == "uncertain" and eff.loc["D1-c0", "verification_status"] == "not_rechecked"
    assert list(VERD.verdict) == ["egg", "not_egg", "egg_bad_boundary", "not_egg"]  # inputs unchanged
