"""Group-aware, deterministic dev/eval split.

Groups are defined by (a) identical SHA-256 and (b) explicit, evidence-backed
decisions in a grouping CSV (columns: group_members, reason). Everything else is a singleton.
"""
from __future__ import annotations

import random

import pandas as pd


class _UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def assign_groups(df: pd.DataFrame, linked_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Add group_id. Images sharing sha256 or listed in linked_pairs (by image_id) share a group."""
    ids = list(df["image_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("image_id values must be unique")
    uf = _UnionFind(ids)
    for _, g in df.groupby("sha256"):
        members = list(g["image_id"])
        for m in members[1:]:
            uf.union(members[0], m)
    known = set(ids)
    for a, b in linked_pairs:
        if a not in known or b not in known:
            raise KeyError(f"linked pair references unknown image_id: {a}, {b}")
        uf.union(a, b)
    out = df.copy()
    out["group_id"] = [uf.find(i) for i in ids]
    return out


def group_split(df: pd.DataFrame, seed: int = 42, eval_fraction: float = 0.2,
                force_dev: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Add 'split' in {dev, eval}. Whole groups go to one side; same seed -> same split.

    Groups are shuffled with a seeded RNG and added to eval while that keeps eval
    at or below the target size; this reaches the target exactly when group sizes allow.
    Any group containing an image_id in force_dev is never eligible for eval (v2: exposure control);
    the shuffle itself is unchanged, so with an empty force_dev the result equals v1.
    """
    if "group_id" not in df:
        raise KeyError("df needs a group_id column; call assign_groups first")
    if len(df) == 0:
        raise ValueError("cannot split an empty dataset")
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be in (0, 1)")
    sizes = df.groupby("group_id").size().sort_index()
    gids = list(sizes.index)
    random.Random(seed).shuffle(gids)
    target = round(len(df) * eval_fraction)
    blocked = set(df.loc[df["image_id"].isin(force_dev), "group_id"])
    unknown = set(force_dev) - set(df["image_id"])
    if unknown:
        raise KeyError(f"force_dev references unknown image_id: {sorted(unknown)}")
    eval_groups, n_eval = set(), 0
    for g in gids:
        if g in blocked:
            continue
        if n_eval + sizes[g] <= target:
            eval_groups.add(g)
            n_eval += sizes[g]
    out = df.copy()
    out["split"] = ["eval" if g in eval_groups else "dev" for g in out["group_id"]]
    return out


def leakage_report(df: pd.DataFrame) -> dict:
    """Counts of groups and sha256 values that appear in both splits (both should be 0)."""
    cross_groups = df.groupby("group_id")["split"].nunique().gt(1).sum()
    cross_hash = df.groupby("sha256")["split"].nunique().gt(1).sum()
    return {
        "n_dev": int((df.split == "dev").sum()),
        "n_eval": int((df.split == "eval").sum()),
        "groups_crossing": int(cross_groups),
        "hashes_crossing": int(cross_hash),
    }


def load_linked_pairs(decisions_csv) -> list[tuple[str, str]]:
    """Pairs whose decision is 'group' in the evidence file; 'no_group' rows are documentation only."""
    d = pd.read_csv(decisions_csv)
    return [(r.image_a, r.image_b) for r in d.itertuples() if r.decision == "group"]


def load_force_dev(constraints_csv) -> frozenset[str]:
    d = pd.read_csv(constraints_csv)
    return frozenset(d.loc[d.constraint == "force_dev", "image_id"])


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from src.data.inventory import inventory

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_raw")
    ap.add_argument("--decisions", default="data/grouping_decisions.csv")
    ap.add_argument("--constraints", default="data/split_constraints.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/split_manifest.csv")
    a = ap.parse_args()
    inv = inventory(Path(a.root))
    if not inv.readable.all():
        raise SystemExit(f"unreadable images: {list(inv.path[~inv.readable])}")
    sp = group_split(assign_groups(inv, load_linked_pairs(a.decisions)), seed=a.seed,
                     force_dev=load_force_dev(a.constraints) if a.constraints else frozenset())
    sp["split_seed"] = a.seed
    sp["split_version"] = "v2_pilot_exposure_controlled"
    sp[["image_id", "path", "sha256", "original_split", "group_id", "split", "split_seed", "split_version"]].to_csv(a.out, index=False)
    print(leakage_report(sp))
    print(sp.groupby("split").size().to_dict(), "groups:", sp.group_id.nunique())
