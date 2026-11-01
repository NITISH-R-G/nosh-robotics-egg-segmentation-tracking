"""Second-pass verification of candidate verdicts against full-context overlays (development images only).

The original verdict file (data/review/dev_instance_verdicts.csv) is never modified. Each re-checked verdict gets a row in
data/review/verdict_verification.csv with status confirmed | corrected | uncertain, the final verdict, an ambiguity category,
the review id, candidate id, mask hash and a reason. `effective_verdicts` merges the two for analysis. All labels are AI-assisted
visual review and are NOT human-verified ground truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.experiments.review_pool import mask_sha256
from src.segmentation.cache_candidates import read_candidates, unpack_masks

VERDICTS = {"egg", "egg_bad_boundary", "not_egg"}
STATUSES = {"confirmed", "corrected", "uncertain"}
CATEGORIES = {"", "clear_not_egg", "clear_egg", "poor_boundary", "ambiguous_object", "partial_view"}
COLS = ["review_id", "image_id", "candidate_id", "mask_sha256", "original_verdict", "status", "final_verdict",
        "ambiguity_category", "reason", "reviewer"]
REVIEWER = "claude_ai_assisted_second_pass_visual_review (not human-verified)"
S1_GREEN, REVIEW_PALETTE = (0, 200, 0), [(0, 0, 255), (255, 80, 0), (255, 0, 255), (0, 200, 255)]


def verification_set(items: pd.DataFrame, verdicts: pd.DataFrame, prov: pd.DataFrame, already_checked: set[str]) -> pd.DataFrame:
    """Round-2-only items (never drawn on a round-1 sheet) not yet checked, plus every egg_bad_boundary item."""
    cfgs = prov.groupby("review_id").config_name.agg(lambda s: {c.split("_")[0] for c in s})
    round2_only = {r for r, c in cfgs.items() if not (c & {"S0", "S1", "S2", "S3", "S4"})}
    v = verdicts.set_index("review_id").verdict
    pick = [r for r in items.review_id if (r in round2_only and r not in already_checked) or v.get(r) == "egg_bad_boundary"]
    return items[items.review_id.isin(pick)].sort_values(["image_id", "candidate_index"]).reset_index(drop=True)


def render_sheets(todo: pd.DataFrame, s1_ids: dict[str, list[int]], verdicts: pd.DataFrame, cache: Path, data: Path,
                  paths: dict, out: Path, per_sheet: int = 4) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    v = verdicts.set_index("review_id").verdict
    images = sorted(todo.image_id.unique(), key=lambda s: int(s.split("_")[-1]))
    index = []
    for s in range(0, len(images), per_sheet):
        panels = []
        for img_id in images[s:s + per_sheet]:
            masks = unpack_masks(cache / "masks" / f"{img_id}.npz")
            img = cv2.imdecode(np.fromfile(str(data / paths[img_id]), np.uint8), cv2.IMREAD_COLOR)
            for i in s1_ids.get(img_id, []):
                cs, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, S1_GREEN, 1)  # thin green: S1 selection (context)
            legend = []
            for k, r in enumerate(todo[todo.image_id == img_id].itertuples()):
                col = REVIEW_PALETTE[k % len(REVIEW_PALETTE)]
                cs, _ = cv2.findContours(masks[r.candidate_index].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, col, 3)
                ys, xs = np.nonzero(masks[r.candidate_index])
                tag = f"c{r.candidate_index}"
                cv2.putText(img, tag, (int(xs.mean()) - 20, int(ys.mean())), 0, 0.9, (0, 0, 0), 5)
                cv2.putText(img, tag, (int(xs.mean()) - 20, int(ys.mean())), 0, 0.9, col, 2)
                legend.append((tag, v.get(r.review_id, ""), col))
            cv2.rectangle(img, (0, 0), (640, 30 + 24 * len(legend)), (0, 0, 0), -1)
            cv2.putText(img, f"img {img_id.split('_')[-1]}  green=S1 selected", (6, 22), 0, 0.65, (0, 255, 255), 2)
            for j, (tag, verdict, col) in enumerate(legend):
                cv2.putText(img, f"{tag}: current verdict = {verdict}", (6, 46 + 24 * j), 0, 0.6, col, 2)
            panels.append(img)
            index.append({"sheet": f"verify_{s // per_sheet:02d}.jpg", "image_id": img_id})
        while len(panels) < per_sheet:
            panels.append(np.zeros((640, 640, 3), np.uint8))
        rows = [np.hstack(panels[r * 2:r * 2 + 2]) for r in range(per_sheet // 2)]
        cv2.imwrite(str(out / f"verify_{s // per_sheet:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    idx = pd.DataFrame(index)
    idx.to_csv(out / "sheet_index.csv", index=False)
    return idx


def record_verification(path: Path, items: pd.DataFrame, verdicts: pd.DataFrame, entries: dict[str, tuple]) -> int:
    """entries: review_id -> (status, final_verdict, ambiguity_category, reason). Append-only; one row per review id."""
    it = items.set_index("review_id")
    v = verdicts.set_index("review_id").verdict
    old = pd.read_csv(path, dtype=str, keep_default_na=False) if Path(path).exists() else pd.DataFrame(columns=COLS)
    rows = []
    for rid, (status, final, cat, reason) in entries.items():
        if rid not in it.index:
            raise KeyError(f"unknown review_id {rid}")
        if rid in set(old.review_id):
            raise ValueError(f"{rid} already verified; verification is append-only")
        if status not in STATUSES or final not in VERDICTS or cat not in CATEGORIES:
            raise ValueError(f"{rid}: invalid status/final verdict/category ({status}, {final}, {cat})")
        if not str(reason).strip():
            raise ValueError(f"{rid}: a reason is required")
        orig = v.get(rid, "")
        if orig == "":
            raise ValueError(f"{rid}: no original verdict to verify")
        if status == "confirmed" and final != orig:
            raise ValueError(f"{rid}: confirmed requires final == original ({orig})")
        if status == "corrected" and final == orig:
            raise ValueError(f"{rid}: corrected requires a different verdict than the original ({orig})")
        r = it.loc[rid]
        rows.append({"review_id": rid, "image_id": r.image_id, "candidate_id": r.canonical_candidate_id,
                     "mask_sha256": r.mask_sha256, "original_verdict": orig, "status": status, "final_verdict": final,
                     "ambiguity_category": cat, "reason": reason, "reviewer": REVIEWER})
    pd.concat([old, pd.DataFrame(rows, columns=COLS)], ignore_index=True).to_csv(path, index=False)
    return len(rows)


def effective_verdicts(verdicts: pd.DataFrame, verification: pd.DataFrame) -> pd.DataFrame:
    """Original verdicts with second-pass corrections applied; adds status and ambiguity columns. Originals stay untouched."""
    ver = verification.set_index("review_id")
    out = verdicts.copy()
    out["original_verdict"] = out.verdict
    out["verification_status"] = out.review_id.map(ver.status).fillna("not_rechecked")
    out["ambiguity_category"] = out.review_id.map(ver.ambiguity_category).fillna("")
    fin = out.review_id.map(ver.final_verdict)
    out["verdict"] = fin.where(fin.notna(), out.verdict)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/sam_candidates")
    ap.add_argument("--data", default="data_raw")
    ap.add_argument("--split", default="data/split_manifest.csv")
    ap.add_argument("--items", default="outputs/review_instances/review_items.csv")
    ap.add_argument("--prov", default="outputs/review_instances/review_provenance.csv")
    ap.add_argument("--verdicts", default="data/review/dev_instance_verdicts.csv")
    ap.add_argument("--out", default="outputs/review_instances/verification_sheets")
    ap.add_argument("--already-checked", nargs="*", default=["D141-c11", "D144-c7", "D149-c2", "D135-c41"])
    a = ap.parse_args()
    sp = pd.read_csv(a.split)
    dev = set(sp.loc[sp["split"] == "dev", "image_id"])
    items, prov = pd.read_csv(a.items), pd.read_csv(a.prov)
    if not set(items.image_id) <= dev:
        raise ValueError("review items contain non-dev images")
    verdicts = pd.read_csv(a.verdicts, dtype=str, keep_default_na=False)
    todo = verification_set(items, verdicts, prov, set(a.already_checked))
    s1 = {}
    for img, g in prov[prov.config_name == "S1_conservative"].groupby("image_id"):
        s1[img] = sorted(int(c.split("#")[1]) for c in g.candidate_id.unique())
    idx = render_sheets(todo, s1, verdicts, Path(a.cache), Path(a.data), dict(zip(sp.image_id, sp.path)), Path(a.out))
    todo[["review_id", "image_id", "canonical_candidate_id", "mask_sha256"]].to_csv(Path(a.out) / "to_verify.csv", index=False)
    print(len(todo), "items on", idx.image_id.nunique(), "images,", idx.sheet.nunique(), "sheets ->", a.out)


if __name__ == "__main__":
    main()
