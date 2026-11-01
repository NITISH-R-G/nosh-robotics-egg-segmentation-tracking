import pandas as pd
import pytest

from src.data.splitting import assign_groups, group_split, leakage_report


def _df(n=50, dup_hash_pairs=((0, 1),)):
    df = pd.DataFrame({"image_id": [f"i{k}" for k in range(n)], "sha256": [f"h{k}" for k in range(n)]})
    for a, b in dup_hash_pairs:
        df.loc[b, "sha256"] = df.loc[a, "sha256"]
    return df


def test_exact_duplicates_and_linked_pairs_share_group():
    g = assign_groups(_df(), [("i5", "i6"), ("i6", "i7")])
    assert g.group_id[0] == g.group_id[1]
    assert len({g.group_id[5], g.group_id[6], g.group_id[7]}) == 1


def test_unknown_linked_pair_raises():
    with pytest.raises(KeyError):
        assign_groups(_df(), [("i5", "zzz")])


@pytest.mark.parametrize("seed", range(20))
def test_groups_never_cross_split(seed):
    pairs = [(f"i{k}", f"i{k+1}") for k in range(2, 40, 4)]
    g = group_split(assign_groups(_df(100, ((0, 1), (50, 51))), pairs), seed=seed)
    rep = leakage_report(g)
    assert rep["groups_crossing"] == 0 and rep["hashes_crossing"] == 0


def test_ratio_and_reproducibility():
    g = assign_groups(_df(110), [("i1", "i2"), ("i3", "i4")])
    a, b = group_split(g, seed=42), group_split(g, seed=42)
    assert (a.split == b.split).all()
    assert (a.split == "eval").sum() == 22 and (a.split == "dev").sum() == 88
    assert not (a.split == group_split(g, seed=7).split).all()


def test_empty_and_bad_args():
    with pytest.raises(ValueError):
        group_split(pd.DataFrame({"group_id": [], "sha256": []}))
    with pytest.raises(KeyError):
        group_split(_df())
    with pytest.raises(ValueError):
        group_split(assign_groups(_df(), []), eval_fraction=1.5)


def test_decisions_file_only_links_confirmed_pairs():
    from pathlib import Path

    from src.data.splitting import load_linked_pairs

    pairs = load_linked_pairs(Path(__file__).parents[1] / "data" / "grouping_decisions.csv")
    assert sorted(map(sorted, pairs)) == [["not_damaged_156", "not_damaged_163"], ["not_damaged_27", "not_damaged_78"]]


def test_force_dev_groups_never_reach_eval_and_groups_stay_together():
    g = assign_groups(_df(110), [("i1", "i2"), ("i3", "i4")])
    forced = frozenset({"i1", "i3", "i10", "i11"})
    for seed in range(30):
        s = group_split(g, seed=seed, force_dev=forced)
        assert leakage_report(s)["groups_crossing"] == 0
        assert (s[s.image_id.isin(["i1", "i2", "i3", "i4", "i10", "i11"])].split == "dev").all()
        assert (s.split == "eval").sum() == 22


def test_force_dev_empty_equals_v1_and_unknown_raises():
    g = assign_groups(_df(110), [("i1", "i2")])
    assert (group_split(g, seed=42).split == group_split(g, seed=42, force_dev=frozenset()).split).all()
    with pytest.raises(KeyError):
        group_split(g, force_dev=frozenset({"nope"}))


def test_real_manifest_v2_properties():
    """data/split_manifest.csv: pilot-exposed and duplicate groups are dev, ratio 88:22, no crossing."""
    from pathlib import Path

    root = Path(__file__).parents[1] / "data"
    m = pd.read_csv(root / "split_manifest.csv")
    old = pd.read_csv(root / "split_manifest_v1_contaminated.csv")
    forced = set(pd.read_csv(root / "split_constraints.csv").image_id)
    assert (m[m.image_id.isin(forced)].split == "dev").all()
    assert leakage_report(m)["groups_crossing"] == 0 and leakage_report(m)["hashes_crossing"] == 0
    assert (m.split == "eval").sum() == 22 and len(m) == 110
    assert set(old[old.split == "eval"].image_id) & forced  # v1 really was contaminated
