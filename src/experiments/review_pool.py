"""Traceable manual-review workflow for cached SAM candidates (development images only).

Inputs: per-image selector outputs (experiments/selector_*_per_image.csv) that list which cached
candidates each configuration selected. Nothing here decides whether a candidate is an egg.

Outputs under --out:
  review_items.csv        one row per visually reviewed mask: review_id, image_id, canonical candidate,
                          mask_sha256, area and measured features
  review_provenance.csv   every (review_id, config_name, experiment_id, code_version, candidate_id) link;
                          identical masks (equal mask_sha256) share one item, similar masks never merge
  sheets/sheet_XXX.jpg    2x2 native-resolution (640 px) panels with outline + review ID per candidate
  sheets/sheet_index.csv  which images are on which sheet
data/review/dev_instance_verdicts.csv is created as a BLANK template (verdict/uncertainty/note empty).

review_id = D<image number>-c<lowest cached candidate index sharing the mask>; it is tied to the
immutable cache, so it does not change when more configurations are added later.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.segmentation.cache_candidates import read_candidates, unpack_masks

PALETTE = [(0, 0, 255), (255, 80, 0), (0, 200, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255)]
FEATURES = ["area", "predicted_iou", "stability_score", "solidity", "aspect", "ellipse_iou", "touches_border"]
VERDICT_COLS = ["review_id", "image_id", "verdict", "uncertainty", "note", "reviewer"]


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(mask, bool).ravel()).tobytes()).hexdigest()


def verify_identity(cands: pd.DataFrame, masks: np.ndarray, img_id: str, idx: int) -> None:
    """The cached mask at position idx must be the very candidate the manifest describes under that id.

    Raises on an out-of-range index, a missing manifest row, or any disagreement of area, bounding box or centroid,
    so a wrong index can never be silently relabelled as another candidate."""
    if not 0 <= idx < len(masks):
        raise IndexError(f"{img_id}#{idx}: index outside the {len(masks)} cached masks")
    rows = cands[(cands.image_id == img_id) & (cands["index"] == idx)]
    if len(rows) != 1:
        raise KeyError(f"{img_id}#{idx}: expected one manifest row, found {len(rows)}")
    rec, m = rows.iloc[0], masks[idx]
    ys, xs = np.nonzero(m)
    ok = (int(m.sum()) == int(rec["area"]) and len(xs) > 0
          and abs(rec["bbox_xywh"][0] - xs.min()) <= 1 and abs(rec["bbox_xywh"][1] - ys.min()) <= 1
          and abs(rec["centroid_xy"][0] - xs.mean()) < 1e-6 and abs(rec["centroid_xy"][1] - ys.mean()) < 1e-6)
    if not ok:
        raise ValueError(f"{img_id}#{idx}: cached mask disagrees with its manifest record (area/bbox/centroid); "
                         f"refusing to treat it as this candidate")


def build_items(runs_glob: str, cache: Path, split_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sp = pd.read_csv(split_csv)
    dev = set(sp.loc[sp["split"] == "dev", "image_id"])
    files = sorted(glob.glob(runs_glob))
    if not files:
        raise FileNotFoundError(f"no selector runs match {runs_glob}")
    sel = []
    for f in files:
        d = pd.read_csv(f)
        for r in d.itertuples():
            for idx in json.loads(r.selected_idx):
                sel.append({"image_id": r.image_id, "candidate_index": int(idx), "config_name": r.config_name,
                            "experiment_id": r.experiment_id, "code_version": r.code_version})
    sel = pd.DataFrame(sel)
    if sel.empty:
        raise ValueError("selectors chose no candidates; nothing to review")
    bad = set(sel.image_id) - dev
    if bad:
        raise ValueError(f"non-dev images in selector runs (refusing to build review items): {sorted(bad)[:5]}")
    cands = read_candidates(cache, set(sel.image_id))  # only the requested (dev) images are ever parsed
    items, prov = [], []
    for img_id, g in sel.groupby("image_id"):
        masks = unpack_masks(cache / "masks" / f"{img_id}.npz")
        num = img_id.split("_")[-1]
        for idx in g.candidate_index.unique():
            verify_identity(cands, masks, img_id, int(idx))
        by_hash: dict[str, int] = {}
        for idx in sorted(g.candidate_index.unique()):
            by_hash.setdefault(mask_sha256(masks[idx]), idx)  # lowest candidate index is canonical
        for idx in sorted(g.candidate_index.unique()):
            h = mask_sha256(masks[idx])
            canon = by_hash[h]
            rid = f"D{num}-c{canon}"
            row = cands[(cands.image_id == img_id) & (cands["index"] == idx)].iloc[0]
            if idx == canon:
                items.append({"review_id": rid, "image_id": img_id, "canonical_candidate_id": f"{img_id}#{canon}",
                              "candidate_index": canon, "mask_sha256": h, **{k: row[k] for k in FEATURES}})
            for r in g[g.candidate_index == idx].itertuples():
                prov.append({"review_id": rid, "image_id": img_id, "candidate_id": f"{img_id}#{idx}",
                             "mask_sha256": h, "config_name": r.config_name, "experiment_id": r.experiment_id,
                             "code_version": r.code_version})
    return pd.DataFrame(items), pd.DataFrame(prov)


GRAY = (150, 150, 150)


def render_panel(img_id: str, items: pd.DataFrame, cache: Path, data: Path, path: str,
                 context: pd.DataFrame | None = None, tags: dict | None = None) -> np.ndarray:
    """Draw review candidates in the image's own (x, y) frame, without resizing.

    items   = candidates to review (colour, thick, labelled with the configs that selected them when tags is given)
    context = already-reviewed candidates (thin gray, labelled with their verdict). Without context a partial sheet
              silently hides previously selected masks, which made correct selections look missing (audit finding).
    """
    img = cv2.imdecode(np.fromfile(str(data / path), np.uint8), cv2.IMREAD_COLOR)
    masks = unpack_masks(cache / "masks" / f"{img_id}.npz")
    tags = tags or {}

    def draw(r, col, thick, label):
        m = masks[r.candidate_index].astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, col, thick)
        ys, xs = np.nonzero(m)
        x, y = int(xs.mean()), int(ys.mean())
        cv2.putText(img, label, (x - 20, y + 8), 0, 0.7 if thick == 1 else 0.9, (0, 0, 0), 4)
        cv2.putText(img, label, (x - 20, y + 8), 0, 0.7 if thick == 1 else 0.9, col, 2 if thick > 1 else 1)

    if context is not None:
        for r in context.itertuples():
            draw(r, GRAY, 1, f"{r.review_id.split('-')[1]}:{str(r.verdict)[:6]}")
    for k, r in enumerate(items.itertuples()):
        label = r.review_id.split("-")[1] + (f" [{tags[r.review_id]}]" if r.review_id in tags else "")
        draw(r, PALETTE[k % len(PALETTE)], 2, label)
    cv2.rectangle(img, (0, 0), (150, 36), (0, 0, 0), -1)
    cv2.putText(img, f"img {img_id.split('_')[-1]}", (6, 27), 0, 0.9, (0, 255, 255), 2)
    if context is not None:
        cv2.rectangle(img, (150, 0), (640, 36), (0, 0, 0), -1)
        cv2.putText(img, "PARTIAL: color=new, gray=already reviewed", (156, 24), 0, 0.55, (0, 255, 255), 1)
    return img


def update_verdict_template(vp: Path, items: pd.DataFrame) -> int:
    """Create the blank verdict file or append blank rows for review_ids not yet present.

    Existing rows (verdicts, uncertainty, notes) are never modified or removed. Returns rows added.
    """
    blank = pd.DataFrame({"review_id": items.review_id, "image_id": items.image_id, "verdict": "",
                          "uncertainty": "", "note": "", "reviewer": ""})[VERDICT_COLS]
    if not vp.exists():
        vp.parent.mkdir(parents=True, exist_ok=True)
        blank.to_csv(vp, index=False)
        return len(blank)
    old = pd.read_csv(vp, dtype=str, keep_default_na=False)
    if list(old.columns) != VERDICT_COLS:
        raise ValueError(f"verdict file columns {list(old.columns)} != {VERDICT_COLS}; refusing to touch it")
    new = blank[~blank.review_id.isin(set(old.review_id))]
    if len(new):
        pd.concat([old, new], ignore_index=True).to_csv(vp, index=False)
    return len(new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="experiments/selector_*_2026*_per_image.csv")
    ap.add_argument("--cache", default="outputs/sam_candidates")
    ap.add_argument("--data", default="data_raw")
    ap.add_argument("--split", default="data/split_manifest.csv")
    ap.add_argument("--out", default="outputs/review_instances")
    ap.add_argument("--verdicts", default="data/review/dev_instance_verdicts.csv")
    ap.add_argument("--only-unreviewed", action="store_true",
                    help="render sheets only for items without a verdict (full item/provenance CSVs are still written)")
    ap.add_argument("--sheets-dir", default="sheets")
    a = ap.parse_args()
    cache, data, out = Path(a.cache), Path(a.data), Path(a.out)
    items, prov = build_items(a.runs, cache, Path(a.split))
    out.mkdir(parents=True, exist_ok=True)
    items.to_csv(out / "review_items.csv", index=False)
    prov.to_csv(out / "review_provenance.csv", index=False)
    sheet_items, reviewed_items = items, None
    short = prov.groupby("review_id").config_name.agg(lambda s: ",".join(sorted({c.split("_")[0] for c in s})))
    tags = short.to_dict()
    if a.only_unreviewed and Path(a.verdicts).exists():
        vd = pd.read_csv(a.verdicts, dtype=str, keep_default_na=False)
        done = set(vd.loc[vd.verdict != "", "review_id"])
        sheet_items = items[~items.review_id.isin(done)]
        reviewed_items = items[items.review_id.isin(done)].merge(vd[["review_id", "verdict"]], on="review_id")
    sheets = out / a.sheets_dir
    sheets.mkdir(exist_ok=True)
    paths = dict(zip(*[pd.read_csv(a.split)[c] for c in ("image_id", "path")]))
    ids = sorted(sheet_items.image_id.unique(), key=lambda s: int(s.split("_")[-1]))
    index = []
    for s in range(0, len(ids), 4):
        chunk = ids[s : s + 4]
        panels = [render_panel(i, sheet_items[sheet_items.image_id == i].sort_values("candidate_index"), cache, data, paths[i],
                               None if reviewed_items is None else reviewed_items[reviewed_items.image_id == i], tags)
                  for i in chunk]
        while len(panels) < 4:
            panels.append(np.zeros((640, 640, 3), np.uint8))
        sheet = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        name = f"sheet_{s // 4:03d}.jpg"
        cv2.imwrite(str(sheets / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        index += [{"sheet": name, "image_id": i} for i in chunk]
    pd.DataFrame(index, columns=["sheet", "image_id"]).to_csv(sheets / "sheet_index.csv", index=False)
    update_verdict_template(Path(a.verdicts), items)
    print(f"{items.image_id.nunique()} images, {len(items)} review items, {len(prov)} provenance links, "
          f"{len(index) // 4 + (len(index) % 4 > 0)} sheets -> {out}")


if __name__ == "__main__":
    main()
