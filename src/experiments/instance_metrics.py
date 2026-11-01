"""Instance-level precision/recall of selector outputs from manual candidate verdicts (dev only).

A selected candidate counts as a true positive when its verdict is 'egg' (strict) or 'egg' /
'egg_bad_boundary' (lenient). False positives are 'not_egg'. Missed eggs = labelled egg count minus
lenient true positives (never negative; an image where TP exceeds the label is reported, not hidden).
Verdicts are AI-assisted manual review of masks, not pixel ground truth; no IoU is computed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def instance_metrics(provenance: pd.DataFrame, verdicts: pd.DataFrame, counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    v = verdicts.set_index("review_id").verdict
    if (v == "").any():
        raise ValueError(f"{int((v == '').sum())} review items have no verdict")
    p = provenance.assign(verdict=provenance.review_id.map(v))
    if p.verdict.isna().any():
        raise ValueError("provenance rows without a verdict entry")
    # one row per (config, image, candidate): provenance may list the same candidate once per config already
    p = p.drop_duplicates(["config_name", "candidate_id"])
    rows = []
    for cfg, g in p.groupby("config_name"):
        per = g.groupby("image_id").verdict.agg(strict=lambda s: (s == "egg").sum(),
                                                lenient=lambda s: s.isin(["egg", "egg_bad_boundary"]).sum(),
                                                fp=lambda s: (s == "not_egg").sum(), bad=lambda s: (s == "egg_bad_boundary").sum())
        m = counts.set_index("image_id").join(per)
        for c in ("strict", "lenient", "fp", "bad"):
            m[c] = m[c].fillna(0).astype(int)
        m["config_name"] = cfg
        m["missed"] = (m.expected_egg_count - m.lenient).clip(lower=0)
        m["tp_exceeds_label"] = m.lenient > m.expected_egg_count
        rows.append(m.reset_index())
    per_image = pd.concat(rows, ignore_index=True)
    agg = []
    for cfg, m in per_image.groupby("config_name"):
        tp_s, tp_l, fp, fn, n = int(m.strict.sum()), int(m.lenient.sum()), int(m.fp.sum()), int(m.missed.sum()), int(m.expected_egg_count.sum())
        sel = tp_l + fp
        agg.append({"config_name": cfg, "labelled_eggs": n, "selected": sel, "tp_strict": tp_s, "tp_lenient": tp_l,
                    "egg_bad_boundary": int(m.bad.sum()), "false_positives": fp, "missed_eggs": fn,
                    "precision_lenient": round(tp_l / sel, 3) if sel else None,
                    "recall_lenient": round(tp_l / n, 3), "precision_strict": round(tp_s / sel, 3) if sel else None,
                    "recall_strict": round(tp_s / n, 3),
                    "f1_lenient": round(2 * tp_l / (sel + n), 3), "images_with_fp": int((m.fp > 0).sum()),
                    "images_with_missed": int((m.missed > 0).sum()), "tp_exceeds_label_images": int(m.tp_exceeds_label.sum())})
    return pd.DataFrame(agg), per_image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prov", default="outputs/review_instances/review_provenance.csv")
    ap.add_argument("--verdicts", default="data/review/dev_instance_verdicts.csv")
    ap.add_argument("--counts", default="data/review/dev_egg_counts.csv")
    ap.add_argument("--out", default="experiments")
    a = ap.parse_args()
    agg, per = instance_metrics(pd.read_csv(a.prov), pd.read_csv(a.verdicts, dtype=str, keep_default_na=False), pd.read_csv(a.counts))
    out = Path(a.out)
    agg.to_csv(out / "selector_instance_metrics.csv", index=False)
    per.to_csv(out / "selector_instance_per_image.csv", index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
