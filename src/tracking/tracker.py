"""Minimal deterministic multi-object tracker: gated Hungarian association on centroid distance + IoU.

No appearance model and no learned components. A track that loses its detection is kept
"lost" for up to `max_age` steps and can be re-linked to a later detection inside a gate that
grows with the time since it was last seen. Re-appearance after the track died gets a new ID
(counted as fragmentation by the metrics, not hidden).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TrackerConfig:
    base_gate: float = 1.2      # gate radius at age 0, in units of the track's mean box diameter
    gate_growth: float = 0.35   # added per missed step, same units
    max_age: int = 24           # steps a track may stay unmatched before it is closed
    iou_weight: float = 0.5     # cost = (1-w)*dist/gate + w*(1-iou)
    min_hits: int = 1           # detections needed before a track is reported (1 = report at once)
    max_tracks: int = 0         # scene prior (0 = off): if this many tracks are open, an unmatched detection
                                # is linked to the nearest LOST track at any distance instead of starting a new ID

    def sha(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class Track:
    tid: int
    bbox: list
    centroid: list
    hits: int = 1
    age: int = 0                # consecutive missed steps
    first_step: int = 0
    last_step: int = 0
    diam: float = 0.0
    history: list = field(default_factory=list)


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _diam(b) -> float:
    return 0.5 * ((b[2] - b[0]) + (b[3] - b[1]))


class Tracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.tracks: list[Track] = []   # open tracks (active or lost)
        self.closed: list[Track] = []
        self._next = 1

    def update(self, step: int, dets: list[dict]) -> list[dict]:
        """Associate detections (dicts with bbox, centroid) at `step`; return [{id, det_index, ...}] for matched/new."""
        cfg = self.cfg
        T, D = self.tracks, dets
        pairs: list[tuple[int, int]] = []
        if T and D:
            cost = np.full((len(T), len(D)), 1e6)
            for i, t in enumerate(T):
                gate = (cfg.base_gate + cfg.gate_growth * t.age) * t.diam
                for j, d in enumerate(D):
                    dist = float(np.hypot(t.centroid[0] - d["centroid"][0], t.centroid[1] - d["centroid"][1]))
                    if dist <= gate:
                        cost[i, j] = (1 - cfg.iou_weight) * dist / gate + cfg.iou_weight * (1 - _iou(t.bbox, d["bbox"]))
            r, c = linear_sum_assignment(cost)
            pairs = [(i, j) for i, j in zip(r, c) if cost[i, j] < 1e5]
        matched_t = {i for i, _ in pairs}
        matched_d = {j for _, j in pairs}
        out = []
        for i, j in pairs:
            t, d = T[i], D[j]
            t.bbox, t.centroid, t.hits, t.age, t.last_step = d["bbox"], d["centroid"], t.hits + 1, 0, step
            t.diam = 0.8 * t.diam + 0.2 * _diam(d["bbox"])
            t.history.append((step, d["centroid"][0], d["centroid"][1]))
            out.append({"id": t.tid, "det": j, "step": step, "new": False})
        for i, t in enumerate(T):
            if i not in matched_t:
                t.age += 1
        for j, d in enumerate(D):
            if j in matched_d:
                continue
            lost = [i for i, t in enumerate(T) if i not in matched_t and t.age > 0]
            if cfg.max_tracks and len(T) >= cfg.max_tracks and lost:
                i = min(lost, key=lambda i: (np.hypot(T[i].centroid[0] - d["centroid"][0], T[i].centroid[1] - d["centroid"][1]), T[i].tid))
                t = T[i]
                matched_t.add(i)
                t.bbox, t.centroid, t.hits, t.age, t.last_step = d["bbox"], d["centroid"], t.hits + 1, 0, step
                t.history.append((step, d["centroid"][0], d["centroid"][1]))
                out.append({"id": t.tid, "det": j, "step": step, "new": False})
                continue
            t = Track(self._next, d["bbox"], d["centroid"], first_step=step, last_step=step, diam=_diam(d["bbox"]),
                      history=[(step, d["centroid"][0], d["centroid"][1])])
            self._next += 1
            T.append(t)
            out.append({"id": t.tid, "det": j, "step": step, "new": True})
        self.closed += [t for t in T if t.age > cfg.max_age]
        self.tracks = [t for t in T if t.age <= cfg.max_age]
        return sorted(out, key=lambda o: o["id"])

    def all_tracks(self) -> list[Track]:
        return sorted(self.closed + self.tracks, key=lambda t: t.tid)
