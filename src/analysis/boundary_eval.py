"""Blind boundary-quality evaluation of refined masks (qualitative, AI-assisted; no ground-truth IoU).

Protocol: experiments/boundary_refinement_protocol.md. For every case the original and the refined masks are shown as columns
A, B, C in a random order (seeded); the mapping is written to blind_mapping.csv and is not opened until ratings are recorded.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.segmentation.cache_candidates import unpack_masks
from src.segmentation.masks import load_mask_png

KINDS = ["orig", "V1", "V2"]
COLUMNS = ["A", "B", "C"]
LEAKAGE = {"none", "minor", "obvious"}
OCCLUSION = {"plausible", "implausible", "not_applicable"}
RATING_COLS = ["case_id", "column", "rating", "covers_intended_egg", "background_leakage", "occlusion", "note"]
PANEL = 480


def case_ids(cases: pd.DataFrame) -> dict[str, str]:
    return {r.review_id: f"C{k + 1:02d}" for k, r in enumerate(cases.itertuples())}


def _crop_box(masks: list[np.ndarray], margin: int = 30) -> tuple[int, int, int, int]:
    u = np.logical_or.reduce(masks)
    ys, xs = np.nonzero(u)
    h, w = u.shape
    x0, x1, y0, y1 = xs.min() - margin, xs.max() + margin, ys.min() - margin, ys.max() + margin
    side = max(x1 - x0, y1 - y0)  # make the crop square, centred on the union
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0, y0 = int(np.clip(cx - side // 2, 0, max(w - side, 0))), int(np.clip(cy - side // 2, 0, max(h - side, 0)))
    return x0, y0, min(x0 + side, w), min(y0 + side, h)


def draw_panel(img: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop, mc = img[y0:y1, x0:x1], mask[y0:y1, x0:x1].astype(np.uint8)
    s = PANEL / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    mc = cv2.resize(mc, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(crop, cs, -1, (0, 0, 255), 2)  # same colour and width for every column: nothing identifies the mask
    out = np.zeros((PANEL, PANEL, 3), np.uint8)
    out[:crop.shape[0], :crop.shape[1]] = crop
    return out


def build_blind_panels(cases: pd.DataFrame, refined_dir: Path, cache: Path, data: Path, split_csv: Path, out: Path,
                       seed: int = 20261029) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(split_csv).set_index("image_id")
    if not set(cases.image_id) <= set(sp.index[sp["split"] == "dev"]):
        raise ValueError("cases contain non-development images")
    rng = np.random.default_rng(seed)
    cid = case_ids(cases)
    rows, panels = [], []
    for r in cases.itertuples():
        orig = unpack_masks(cache / "masks" / f"{r.image_id}.npz")[int(r.candidate_index)]
        stem = f"{r.image_id}__{int(r.candidate_index)}"
        m = {"orig": orig, "V1": load_mask_png(refined_dir / "V1" / "masks" / f"{stem}.png", orig.shape).astype(bool),
             "V2": load_mask_png(refined_dir / "V2" / "masks" / f"{stem}.png", orig.shape).astype(bool)}
        for k, v in m.items():
            if not v.any():
                raise ValueError(f"{r.review_id}: empty {k} mask")
        order = list(rng.permutation(KINDS))
        img = cv2.imdecode(np.fromfile(str(data / sp.loc[r.image_id, "path"]), np.uint8), cv2.IMREAD_COLOR)
        box = _crop_box(list(m.values()))
        cols = []
        for col, kind in zip(COLUMNS, order):
            p = draw_panel(img, m[kind], box)
            cv2.putText(p, col, (10, 40), 0, 1.4, (0, 0, 0), 6)
            cv2.putText(p, col, (10, 40), 0, 1.4, (0, 255, 255), 3)
            cols.append(p)
        row = np.hstack(cols)
        cv2.putText(row, f"case {cid[r.review_id]}  img {r.image_id.split('_')[-1]}", (10, PANEL - 14), 0, 0.8, (0, 0, 0), 5)
        cv2.putText(row, f"case {cid[r.review_id]}  img {r.image_id.split('_')[-1]}", (10, PANEL - 14), 0, 0.8, (255, 255, 255), 2)
        panels.append(row)
        rows.append({"case_id": cid[r.review_id], "review_id": r.review_id, "image_id": r.image_id,
                     **{f"column_{c}": k for c, k in zip(COLUMNS, order)}})
    sheets = []
    for s in range(0, len(panels), 2):
        chunk = panels[s:s + 2]
        while len(chunk) < 2:
            chunk.append(np.zeros_like(panels[0]))
        name = f"blind_{s // 2:02d}.jpg"
        cv2.imwrite(str(out / name), np.vstack(chunk), [cv2.IMWRITE_JPEG_QUALITY, 92])
        sheets += [{"sheet": name, "case_id": rows[s + k]["case_id"]} for k in range(len(panels[s:s + 2]))]
    pd.DataFrame(sheets).to_csv(out / "blind_sheet_index.csv", index=False)
    mapping = pd.DataFrame(rows)
    mapping.to_csv(out / "blind_mapping.csv", index=False)  # do not open before all ratings are recorded
    return pd.DataFrame(sheets)


def record_ratings(path: Path, entries: dict[tuple[str, str], tuple], valid_cases: set[str]) -> int:
    """entries: (case_id, column) -> (rating, covers, leakage, occlusion, note). Append-only; every (case, column) once."""
    old = pd.read_csv(path, dtype=str, keep_default_na=False) if Path(path).exists() else pd.DataFrame(columns=RATING_COLS)
    seen = set(zip(old.case_id, old.column))
    rows = []
    for (cid, col), (rating, covers, leak, occl, note) in entries.items():
        if cid not in valid_cases or col not in COLUMNS:
            raise KeyError(f"unknown case/column {cid}/{col}")
        if (cid, col) in seen:
            raise ValueError(f"{cid}/{col} already rated; ratings are append-only")
        if rating not in (1, 2, 3) or covers not in ("yes", "no") or leak not in LEAKAGE or occl not in OCCLUSION:
            raise ValueError(f"{cid}/{col}: invalid rating fields {(rating, covers, leak, occl)}")
        rows.append({"case_id": cid, "column": col, "rating": str(rating), "covers_intended_egg": covers,
                     "background_leakage": leak, "occlusion": occl, "note": note})
    pd.concat([old, pd.DataFrame(rows, columns=RATING_COLS)], ignore_index=True).to_csv(path, index=False)
    return len(rows)


def resolve(ratings: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Unblind: one row per (case, kind) with its rating and flags. Every case must have all three columns rated."""
    long = mapping.melt(id_vars=["case_id", "review_id", "image_id"], value_vars=[f"column_{c}" for c in COLUMNS],
                        var_name="column", value_name="kind")
    long["column"] = long.column.str[-1]
    r = long.merge(ratings, on=["case_id", "column"], how="left")
    if r.rating.isna().any():
        raise ValueError(f"unrated columns: {r[r.rating.isna()][['case_id', 'column']].values.tolist()[:5]}")
    r["rating"] = r.rating.astype(int)
    return r


def classify(resolved: pd.DataFrame) -> pd.DataFrame:
    """improved / comparable / degraded of each refined variant relative to the original (rating difference >= 1)."""
    piv = resolved.pivot(index="case_id", columns="kind", values="rating")
    out = []
    for v in ("V1", "V2"):
        d = piv[v] - piv["orig"]
        for cid, delta in d.items():
            out.append({"variant": v, "case_id": cid, "orig_rating": int(piv.loc[cid, "orig"]), "refined_rating": int(piv.loc[cid, v]),
                        "outcome": "improved" if delta >= 1 else "degraded" if delta <= -1 else "comparable"})
    return pd.DataFrame(out)


def evaluate_criteria(resolved: pd.DataFrame, classified: pd.DataFrame, targets: set[str], controls: set[str]) -> pd.DataFrame:
    """Protocol criteria A1-A4 per variant. targets = the 10 certain target case ids; controls = the 6 control case ids."""
    rows = []
    for v in ("V1", "V2"):
        c = classified[classified.variant == v].set_index("case_id")
        judged = resolved[(resolved.kind == v) & resolved.case_id.isin(targets | controls)]
        t, k = c.loc[sorted(targets)], c.loc[sorted(controls)]
        a1 = int((t.outcome == "improved").sum()) >= 6 and int((t.outcome == "degraded").sum()) <= 1
        a2 = int((k.outcome == "degraded").sum()) == 0
        a3 = int((judged.background_leakage == "obvious").sum()) <= 1
        a4 = int((judged.covers_intended_egg == "yes").sum()) >= 0.9 * len(judged)
        rows.append({"variant": v, "targets_improved": int((t.outcome == "improved").sum()), "targets_degraded": int((t.outcome == "degraded").sum()),
                     "controls_degraded": int((k.outcome == "degraded").sum()), "obvious_leakage": int((judged.background_leakage == "obvious").sum()),
                     "covers_egg_yes": int((judged.covers_intended_egg == "yes").sum()), "judged_masks": len(judged),
                     "A1_targets": a1, "A2_controls": a2, "A3_leakage": a3, "A4_coverage": a4, "ACCEPTED": bool(a1 and a2 and a3 and a4)})
    return pd.DataFrame(rows)
