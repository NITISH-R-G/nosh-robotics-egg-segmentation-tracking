"""Dev-only end-to-end diagnostic of selector outputs and review-sheet provenance.

For each requested image it verifies, with exact candidate IDs and mask hashes:
  I1 cached mask bytes agree with the candidate manifest (area, bbox, centroid)
  I2 review item mask hash and candidate id agree with the cache
  I3 stored per-image selection equals a live re-selection from the cache
  I4 the saved selector union mask equals the union of the cached masks of the selected ids
  I5 coordinate system: the SAM point prompt lies inside its mask, and rendered contours coincide with the mask boundary
and classifies each expected egg: covered by a selected candidate / selection failure (a reviewed egg candidate exists that
the selector did not choose) / proposal failure (no reviewed egg candidate exists) . Verdicts come from the AI-assisted
candidate review and are not pixel ground truth. Evaluation images are rejected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.experiments.review_pool import mask_sha256
from src.experiments.selector_experiments import CONFIGS, load_dev
from src.segmentation.cache_candidates import unpack_masks
from src.segmentation.masks import load_mask_png
from src.segmentation.selector import SelectorConfig, select_from_features


def assert_dev(image_ids: list[str], split_csv: Path) -> None:
    sp = pd.read_csv(split_csv).set_index("image_id")["split"]
    unknown = [i for i in image_ids if i not in sp.index]
    if unknown:
        raise ValueError(f"unknown image ids: {unknown}")
    not_dev = [i for i in image_ids if sp[i] != "dev"]
    if not_dev:
        raise ValueError(f"refusing to diagnose non-development images: {not_dev}")


def explain_gate(row: pd.Series, area_frac: float, cfg: SelectorConfig) -> list[dict]:
    """Every condition of the primary gate with the measured value, threshold and pass/fail."""
    e_min = cfg.min_ellipse_iou
    if row["touches_border"] and cfg.border_min_ellipse_iou is not None:
        e_min = cfg.border_min_ellipse_iou
    aspect = row["aspect"]
    rules = [("area_frac>=min", area_frac, cfg.min_area_frac, area_frac >= cfg.min_area_frac),
             ("area_frac<=max", area_frac, cfg.max_area_frac, area_frac <= cfg.max_area_frac),
             ("aspect<=max", aspect, cfg.max_aspect, bool(pd.notna(aspect) and aspect <= cfg.max_aspect)),
             ("solidity>=min", row["solidity"], cfg.min_solidity, row["solidity"] >= cfg.min_solidity),
             ("ellipse_iou>=min", row["ellipse_iou"], e_min, row["ellipse_iou"] >= e_min),
             ("pred_iou>=min", row["predicted_iou"], cfg.min_pred_iou, row["predicted_iou"] >= cfg.min_pred_iou),
             ("stability>=min", row["stability_score"], cfg.min_stability, row["stability_score"] >= cfg.min_stability)]
    return [{"rule": n, "value": round(float(v), 4) if pd.notna(v) else None, "threshold": t, "pass": bool(ok)} for n, v, t, ok in rules]


def contour_matches_mask(mask: np.ndarray, drawn: np.ndarray | None = None, tol: float = 1.5) -> bool:
    """Contours of `drawn` (default: the mask itself) must lie on the boundary of `mask` in the same (x, y) frame.

    Passing a transposed or shifted `drawn` mask makes this return False, which is what the tests rely on."""
    drawn = mask if drawn is None else drawn
    if drawn.shape != mask.shape:
        return False
    # Pad with zeros: masks touching the image border must still have a boundary there (the default border mode of
    # the morphology call replicates edge pixels and would report no edge, a false alarm found during this audit).
    m = np.pad(mask.astype(np.uint8), 1)
    d = np.pad(drawn.astype(np.uint8), 1)
    cs, _ = cv2.findContours(d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return not m.any() and not d.any()
    edge = (m > 0) & (cv2.erode(m, np.ones((3, 3), np.uint8), borderValue=0) == 0)
    if not edge.any():
        return False
    dist = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
    pts = np.concatenate([c.reshape(-1, 2) for c in cs])  # (x, y) in padded frame
    return bool(dist[pts[:, 1], pts[:, 0]].max() <= tol)


def prompt_point_distance(mask: np.ndarray, x: float, y: float) -> float:
    """Distance in px from the SAM prompt point to the mask (0 if inside). SAM does not guarantee the point lies
    inside the returned mask (about 10% of dev candidates are 1-5 px outside), but a transposed or flipped
    mask would put most points far away."""
    xi, yi = int(round(x)), int(round(y))
    if mask[yi, xi]:
        return 0.0
    return float(cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)[yi, xi])


def verify_candidate(rec: pd.Series, mask: np.ndarray, item_hash: str | None = None) -> dict:
    ys, xs = np.nonzero(mask)
    bx, by, bw, bh = rec["bbox_xywh"]
    px, py = rec["point_coords_xy"]
    h = mask_sha256(mask)
    checks = {
        "I1_area": int(mask.sum()) == int(rec["area"]),
        "I1_bbox": bool(len(xs)) and abs(bx - xs.min()) <= 1 and abs(by - ys.min()) <= 1
                   and abs(bw - (xs.max() - xs.min() + 1)) <= 1 and abs(bh - (ys.max() - ys.min() + 1)) <= 1,
        "I1_centroid": bool(len(xs)) and abs(rec["centroid_xy"][0] - xs.mean()) < 1e-6 and abs(rec["centroid_xy"][1] - ys.mean()) < 1e-6,
        "I2_hash_matches_review_item": item_hash is None or item_hash == h,
        "I5_point_prompt_within_5px_of_mask": bool(0 <= round(py) < mask.shape[0] and 0 <= round(px) < mask.shape[1]
                                                   and prompt_point_distance(mask, px, py) <= 5.0),
        "I5_contour_on_mask_boundary": contour_matches_mask(mask),
    }
    return {"mask_sha256": h, **checks}


def classify_misses(expected: int, selected_verdicts: list[str], unselected_egg_ids: list[str]) -> str:
    covered = sum(v in ("egg", "egg_bad_boundary") for v in selected_verdicts)
    if covered >= expected:
        return "covered"
    return "B_selection_failure" if unselected_egg_ids else "A_proposal_failure_or_unreviewed"


def diagnose_image(img_id: str, cache: Path, data: Path, cands: pd.DataFrame, counts: pd.DataFrame, items: pd.DataFrame,
                   prov: pd.DataFrame, verdicts: pd.DataFrame, s1_run_csv: Path, s1_masks_dir: Path, split_csv: Path,
                   cfg: SelectorConfig, out: Path) -> dict:
    dfi = cands[cands.image_id == img_id].set_index("index", drop=False)
    masks = unpack_masks(cache / "masks" / f"{img_id}.npz")
    h, w = masks.shape[1:]
    exp_row = counts[counts.image_id == img_id].iloc[0]
    # I3 live re-selection vs stored selection
    live, status = select_from_features(dfi, lambda i: masks[i], h * w, cfg)
    stored = json.loads(pd.read_csv(s1_run_csv).set_index("image_id").loc[img_id, "selected_idx"])
    i3 = sorted(live) == sorted(stored)
    # I4 saved union
    saved = load_mask_png(s1_masks_dir / f"{img_id}.png", (h, w))
    union = np.zeros((h, w), np.uint8)
    for i in live:
        union |= masks[i].astype(np.uint8)
    i4 = bool((saved == union).all())
    v = verdicts.set_index("review_id").verdict
    it = items[items.image_id == img_id].set_index("candidate_index")
    review_of = {}
    for i in it.index:
        review_of[int(i)] = it.loc[i, "review_id"]

    def info(i: int, why: str) -> dict:
        rec = dfi.loc[i]
        item_hash = it.loc[i, "mask_sha256"] if i in it.index else None
        ver = verify_candidate(rec, masks[i], item_hash)
        rid = review_of.get(i)
        cfgs = sorted(set(prov[prov.review_id == rid].config_name)) if rid else []
        return {"candidate_id": f"{img_id}#{i}", "index": int(i), "review_id": rid, "verdict": v.get(rid, "") if rid else "",
                "configs_that_selected": cfgs, "area": int(rec["area"]), "area_frac": round(float(rec["area"]) / (h * w), 4),
                "bbox_xywh": rec["bbox_xywh"], "centroid_xy": [round(x, 1) for x in rec["centroid_xy"]],
                "predicted_iou": round(float(rec["predicted_iou"]), 3), "stability": round(float(rec["stability_score"]), 3),
                "solidity": round(float(rec["solidity"]), 3), "ellipse_iou": round(float(rec["ellipse_iou"]), 3),
                "touches_border": bool(rec["touches_border"]), "why": why,
                "S1_gate": explain_gate(rec, float(rec["area"]) / (h * w), cfg), **ver}

    sel = [info(i, "selected by S1") for i in live]
    # reviewed egg candidates that S1 did NOT select, plus every other reviewed candidate for context
    others = []
    for i, rid in sorted(review_of.items()):
        if i in live:
            continue
        failed = [r["rule"] for r in explain_gate(dfi.loc[i], float(dfi.loc[i, "area"]) / (h * w), cfg) if not r["pass"]]
        why = ("passes the S1 gate but suppressed by NMS or ranking" if not failed else "rejected by S1 gate: " + ", ".join(failed))
        others.append(info(i, why))
    sel_verdicts = [s["verdict"] for s in sel]
    unsel_eggs = [o["candidate_id"] for o in others if o["verdict"] in ("egg", "egg_bad_boundary")]
    klass = classify_misses(int(exp_row.expected_egg_count), sel_verdicts, unsel_eggs)
    integrity = all(k for s in sel + others for kk, k in s.items() if kk.startswith(("I1", "I2", "I5"))) and i3 and i4

    # ---- corrected diagnostic overlay: original | S1 selected | other reviewed candidates | cached S1 masks as binary
    img = cv2.imdecode(np.fromfile(str(data / pd.read_csv(split_csv).set_index("image_id").loc[img_id, "path"]), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (h, w), "image/mask shape mismatch"

    def draw(idxs, color, label_fmt):
        o = img.copy()
        for i in idxs:
            cs, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(o, cs, -1, color, 3)
            ys, xs = np.nonzero(masks[i])
            cv2.putText(o, label_fmt(i), (int(xs.mean()) - 30, int(ys.mean())), 0, 0.8, (0, 0, 0), 5)
            cv2.putText(o, label_fmt(i), (int(xs.mean()) - 30, int(ys.mean())), 0, 0.8, color, 2)
        return o

    p0 = img.copy()
    p1 = draw(live, (0, 200, 0), lambda i: f"S1 c{i}")
    p2 = draw([int(o["index"]) for o in others], (0, 0, 255), lambda i: f"c{i}:{(v.get(review_of[i], '') or '?')[:6]}")
    p3 = np.zeros((h, w, 3), np.uint8)
    for i in live:
        p3[masks[i]] = (255, 255, 255)
    for k, p in enumerate((p0, p1, p2, p3)):
        cv2.putText(p, ["original", "S1 selected (green) - cached masks", "other reviewed candidates (red)", "binary: union of S1 cached masks"][k],
                    (8, 24), 0, 0.7, (0, 255, 255), 2)
    sheet = np.vstack([np.hstack([p0, p1]), np.hstack([p2, p3])])
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{img_id}.jpg"), cv2.resize(sheet, None, fx=0.75, fy=0.75), [cv2.IMWRITE_JPEG_QUALITY, 88])
    rec = {"image_id": img_id, "split": "dev", "expected_egg_count": int(exp_row.expected_egg_count),
           "difficulty_tag": "" if pd.isna(exp_row.difficulty_tag) else exp_row.difficulty_tag,
           "annotation_note": "AI-assisted, unverified by a human", "I3_live_selection_equals_stored": i3,
           "I4_saved_union_equals_cached_union": i4, "integrity_all_ok": bool(integrity), "classification": klass,
           "S1_status": status, "S1_selected": sel, "other_reviewed_candidates": others}
    (out / f"{img_id}.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", type=int, required=True, help="image numbers, e.g. 141 144")
    ap.add_argument("--cache", default="outputs/sam_candidates")
    ap.add_argument("--data", default="data_raw")
    ap.add_argument("--split", default="data/split_manifest.csv")
    ap.add_argument("--counts", default="data/review/dev_egg_counts.csv")
    ap.add_argument("--config", default="S1_conservative")
    ap.add_argument("--s1-run", required=True, help="per-image CSV of the S1 run")
    ap.add_argument("--s1-masks", default="outputs/selector_dev/S1_conservative/masks")
    ap.add_argument("--out", default="outputs/diagnostics")
    a = ap.parse_args()
    ids = [f"not_damaged_{n}" for n in a.images]
    assert_dev(ids, Path(a.split))  # before anything else is loaded
    cache = Path(a.cache)
    cands, counts = load_dev(cache, Path(a.split), Path(a.counts))
    items = pd.read_csv("outputs/review_instances/review_items.csv")
    prov = pd.read_csv("outputs/review_instances/review_provenance.csv")
    verdicts = pd.read_csv("data/review/dev_instance_verdicts.csv", dtype=str, keep_default_na=False)
    rows = []
    for i in ids:
        r = diagnose_image(i, cache, Path(a.data), cands, counts, items, prov, verdicts, Path(a.s1_run), Path(a.s1_masks),
                           Path(a.split), CONFIGS[a.config], Path(a.out))
        rows.append({"image_id": i, "expected": r["expected_egg_count"], "tag": r["difficulty_tag"],
                     "S1_selected": ",".join(s["candidate_id"].split("#")[1] for s in r["S1_selected"]),
                     "S1_verdicts": ",".join(s["verdict"] for s in r["S1_selected"]),
                     "S1_mask_sha256": ",".join(s["mask_sha256"][:12] for s in r["S1_selected"]),
                     "integrity_ok": r["integrity_all_ok"], "classification": r["classification"]})
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out) / "summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
