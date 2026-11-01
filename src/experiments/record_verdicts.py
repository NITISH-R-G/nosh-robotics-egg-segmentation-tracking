"""Write manual candidate verdicts into data/review/dev_instance_verdicts.csv.

Only blank rows are filled; a row that already has a verdict is never changed (corrections must be
made by an explicit, separately logged edit). Verdicts: egg | egg_bad_boundary | not_egg.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

VERDICTS = {"egg", "egg_bad_boundary", "not_egg"}
UNCERTAINTY = {"", "low", "medium", "high"}


def record_verdicts(path: Path, entries: dict[str, tuple], reviewer: str = "claude_ai_assisted_visual_review") -> int:
    """entries: review_id -> (verdict, uncertainty, note). Returns rows filled."""
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    unknown = set(entries) - set(d.review_id)
    if unknown:
        raise KeyError(f"unknown review_ids: {sorted(unknown)[:5]}")
    n = 0
    for rid, (verdict, unc, note) in entries.items():
        if verdict not in VERDICTS:
            raise ValueError(f"{rid}: invalid verdict {verdict!r}")
        if unc not in UNCERTAINTY:
            raise ValueError(f"{rid}: invalid uncertainty {unc!r}")
        i = d.index[d.review_id == rid][0]
        if d.loc[i, "verdict"] != "":
            raise ValueError(f"{rid}: already has verdict {d.loc[i, 'verdict']!r}; refusing to overwrite")
        d.loc[i, ["verdict", "uncertainty", "note", "reviewer"]] = [verdict, unc, note, reviewer]
        n += 1
    d.to_csv(path, index=False)
    return n
