"""Proxy identity evaluation for the egg video.

Ground truth is a hand-written, single-annotator (AI-assisted) description of which physical egg (Q = the egg
labelled "Raw Egg", P = the other) each detection belongs to, stored as time segments in
data/video/identity_segments.csv. It is NOT independent human ground truth, so the numbers below are a proxy.
Segmentation errors (missed / extra detections) are counted separately from identity errors.
"""
from __future__ import annotations

import csv
from collections import defaultdict

FRAME_W = 1920.0

# identity gaps: the identity is expected visible just before `start` and again from `end` (sample frames)
GAPS = [("Q", 480, 558, "hand lifts and carries Q to the right"),
        ("Q", 648, 750, "hand carries Q back to the far left"),
        ("P", 750, 816, "hand carries P to the right")]


def load_segments(path: str = "data/video/identity_segments.csv") -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["start"], r["end"], r["expected_visible"] = int(r["start"]), int(r["end"]), int(r["expected_visible"])
    return rows


def segment_for(frame: int, segs: list[dict]) -> dict | None:
    for s in segs:
        if s["start"] <= frame < s["end"]:
            return s
    return None


def label_detections(frame: int, xs: list[float], segs: list[dict]) -> list[str | None]:
    """Physical identity (P/Q) for each detection centre x (original pixels), by the segment rule."""
    s = segment_for(frame, segs)
    if s is None:
        return [None] * len(xs)
    rule = s["rule"]
    xn = [x / FRAME_W for x in xs]
    if rule.startswith("ALL="):
        return [rule[4:]] * len(xs)
    if rule.startswith("X<"):
        thr, rest = rule[2:].split("=", 1)
        thr = float(thr)
        q_label, other = rest.split(";")[0], rest.split(";")[1].split("=")[1]
        return [q_label if v < thr else other for v in xn]
    # "L=a;R=b": leftmost -> a, rightmost -> b (single detection: ambiguous -> None)
    left, right = (p.split("=")[1] for p in rule.split(";"))
    if len(xs) != 2:
        return [None] * len(xs)
    order = sorted(range(2), key=lambda i: xs[i])
    lab = [None, None]
    lab[order[0]], lab[order[1]] = left, right
    return lab


def evaluate(rows: list[dict], frames: list[int], segs: list[dict], ambiguous: set[int], subset: set[int] | None = None) -> dict:
    """rows: tracks.csv rows (frame, id, x1..x2). frames: sampled frames. subset: restrict to these frames."""
    by_frame = defaultdict(list)
    for r in rows:
        if r["id"] != "":
            by_frame[int(r["frame"])].append(r)
    used = [f for f in frames if f not in ambiguous and segment_for(f, segs) is not None and (subset is None or f in subset)]
    E = D = covered = missed = extra = 0
    seq = defaultdict(list)          # identity -> [(frame, track id)]
    purity = defaultdict(lambda: defaultdict(int))
    for f in used:
        exp = segment_for(f, segs)["expected_visible"]
        dets = by_frame.get(f, [])
        E += exp
        D += len(dets)
        covered += min(exp, len(dets))
        missed += max(exp - len(dets), 0)
        extra += max(len(dets) - exp, 0)
        xs = [(float(d["x1"]) + float(d["x2"])) / 2 for d in dets]
        for d, lab in zip(dets, label_detections(f, xs, segs)):
            if lab is not None:
                seq[lab].append((f, int(d["id"])))
                purity[int(d["id"])][lab] += 1
    idsw = sum(1 for s in seq.values() for a, b in zip(s, s[1:]) if a[1] != b[1])
    ids_per_identity = {k: sorted({i for _, i in v}) for k, v in seq.items()}
    all_ids = sorted({int(r["id"]) for r in rows if r["id"] != "" and int(r["frame"]) in set(used)})
    gaps = []
    for ident, a, b, why in GAPS:
        before = [i for f, i in seq.get(ident, []) if f < a]
        after = [i for f, i in seq.get(ident, []) if f >= b]
        if (subset is not None and not any(a <= f < b or f < a for f in subset)) or not before or not after:
            gaps.append({"identity": ident, "start": a, "end": b, "why": why, "retained": None})
        else:
            gaps.append({"identity": ident, "start": a, "end": b, "why": why, "retained": before[-1] == after[0],
                         "id_before": before[-1], "id_after": after[0]})
    return {"frames_used": len(used), "expected_eggs": E, "detections": D, "coverage": covered / E if E else float("nan"),
            "missed": missed, "extra": extra, "id_switches": idsw, "ids_per_identity": ids_per_identity,
            "distinct_ids": all_ids, "extra_ids": max(len(all_ids) - len(seq), 0),
            "track_purity": {k: dict(v) for k, v in purity.items()}, "gaps": gaps}
