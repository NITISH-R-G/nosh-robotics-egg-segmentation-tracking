"""Tracker K: constant-velocity Kalman prediction + gated Hungarian association with explicit observed/predicted/uncertain flags.

T0/T1 (`tracker.py`) are unchanged. Everything is deterministic. Predicted boxes are reported separately and never counted as
detections. Partial detections (fragments of an egg under the hand) can update a track with larger measurement noise but can
never start one. The two-egg relink prior of T1 is kept and every such relink is flagged `uncertain` (it is a scene prior, not evidence).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class KConfig:
    sigma_meas: float = 0.08      # measurement std, in mean box diameters (complete detections)
    partial_noise_mult: float = 3.0
    q_acc: float = 0.35           # process (acceleration) std per step, in mean box diameters
    vel_damp: float = 0.85        # per-step velocity damping while predicting
    chi2_gate: float = 16.0       # Mahalanobis^2 gate (2 dof)
    iou_weight: float = 0.3
    max_age: int = 24             # steps without an observation before a track is closed
    max_predicted_display: int = 6  # predicted boxes are only reported for this many missed steps
    max_tracks: int = 2           # two-egg prior (0 = off): relink unmatched complete detection to nearest lost track
    ambiguity_ratio: float = 0.8  # best/second-best cost above this -> 'uncertain'
    frame_w: int = 0              # >0 with frame_h: predicted boxes are kept fully inside the frame (velocity zeroed on the clamped axis)
    frame_h: int = 0

    def sha(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class KTrack:
    tid: int
    x: np.ndarray                 # rows: x, y; columns: position, velocity
    P: np.ndarray                 # 2x2 covariance (position, velocity) shared by both axes
    w: float
    h: float
    first_step: int
    last_step: int
    age: int = 0
    last_obs_xy: tuple = (0.0, 0.0)
    obs_kinds: list = field(default_factory=list)

    @property
    def diam(self) -> float:
        return 0.5 * (self.w + self.h)

    def center(self):
        return float(self.x[0, 0]), float(self.x[1, 0])


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class KalmanTracker:
    def __init__(self, cfg: KConfig):
        self.cfg = cfg
        self.tracks: list[KTrack] = []
        self.closed: list[KTrack] = []
        self._next = 1

    def _predict(self, t: KTrack) -> None:
        d = self.cfg.vel_damp
        F = np.array([[1.0, 1.0], [0.0, d]])
        q = (self.cfg.q_acc * t.diam) ** 2
        Q = q * np.array([[0.25, 0.5], [0.5, 1.0]])
        t.x = t.x @ F.T
        t.P = F @ t.P @ F.T + Q
        if self.cfg.frame_w and self.cfg.frame_h:
            for ax, (size, half) in enumerate(((self.cfg.frame_w, t.w / 2), (self.cfg.frame_h, t.h / 2))):
                lo, hi = half, size - half
                if t.x[ax, 0] < lo or t.x[ax, 0] > hi:
                    t.x[ax, 0] = min(max(t.x[ax, 0], lo), hi)
                    t.x[ax, 1] = 0.0

    def _S(self, t: KTrack, partial: bool) -> float:
        r = self.cfg.sigma_meas * t.diam * (self.cfg.partial_noise_mult if partial else 1.0)
        return float(t.P[0, 0] + r * r)

    def _update(self, t: KTrack, det: dict, step: int, partial: bool) -> None:
        S = self._S(t, partial)
        K = t.P[:, 0] / S
        z = np.array(det["centroid"], float)
        resid = z - t.x[:, 0]
        t.x = t.x + np.outer(resid, K)          # (2 axes) x (position, velocity)
        t.P = t.P - np.outer(K, t.P[0, :])
        b = det["bbox"]
        if not partial:                           # box size is learned only from whole eggs
            t.w = 0.8 * t.w + 0.2 * (b[2] - b[0])
            t.h = 0.8 * t.h + 0.2 * (b[3] - b[1])
        t.age, t.last_step, t.last_obs_xy = 0, step, (float(z[0]), float(z[1]))
        t.obs_kinds.append("partial" if partial else "complete")

    def _reset_to(self, t: KTrack, d: dict, step: int) -> None:
        cfg = self.cfg
        t.x = np.array([[d["centroid"][0], 0.0], [d["centroid"][1], 0.0]])
        t.P = np.diag([(cfg.sigma_meas * t.diam) ** 2, (0.5 * t.diam) ** 2])
        b = d["bbox"]
        t.w, t.h = 0.8 * t.w + 0.2 * (b[2] - b[0]), 0.8 * t.h + 0.2 * (b[3] - b[1])
        t.age, t.last_step, t.last_obs_xy = 0, step, (float(d["centroid"][0]), float(d["centroid"][1]))
        t.obs_kinds.append("complete")

    def update(self, step: int, dets: list[dict]) -> tuple[list[dict], list[dict]]:
        """dets: dicts with bbox, centroid, kind ('complete'|'partial'). Returns (observed rows, predicted rows)."""
        cfg = self.cfg
        T = self.tracks
        for t in T:
            self._predict(t)
        D = dets
        out_obs, out_pred = [], []
        cost = np.full((len(T), len(D)), np.inf)
        for i, t in enumerate(T):
            cx, cy = t.center()
            pb = [cx - t.w / 2, cy - t.h / 2, cx + t.w / 2, cy + t.h / 2]
            for j, d in enumerate(D):
                partial = d["kind"] == "partial"
                d2 = ((d["centroid"][0] - cx) ** 2 + (d["centroid"][1] - cy) ** 2) / self._S(t, partial)
                if d2 <= cfg.chi2_gate:
                    cost[i, j] = d2 / cfg.chi2_gate + cfg.iou_weight * (1 - _iou(pb, d["bbox"]))
        pairs = []
        if T and D:
            work = np.where(np.isfinite(cost), cost, 1e6)
            r, c = linear_sum_assignment(work)
            pairs = [(i, j) for i, j in zip(r, c) if np.isfinite(cost[i, j])]
        matched_t, matched_d = {i for i, _ in pairs}, {j for _, j in pairs}
        for i, j in pairs:
            best = cost[i, j]
            others = [cost[i, k] for k in range(len(D)) if k != j and np.isfinite(cost[i, k])] + \
                     [cost[k, j] for k in range(len(T)) if k != i and np.isfinite(cost[k, j])]
            second = min(others) if others else np.inf
            unc = bool(np.isfinite(second) and second > 1e-9 and best / second > cfg.ambiguity_ratio)
            t, d = T[i], D[j]
            self._update(t, d, step, d["kind"] == "partial")
            out_obs.append({"id": t.tid, "det": j, "new": False, "uncertain": unc, "reason": "ambiguous association" if unc else ""})
        for i, t in enumerate(T):
            if i not in matched_t:
                t.age += 1
        for j, d in enumerate(D):
            if j in matched_d:
                continue
            if d["kind"] == "partial":
                out_obs.append({"id": None, "det": j, "new": False, "uncertain": False, "reason": "unassigned partial"})
                continue
            lost = [i for i, t in enumerate(T) if i not in matched_t and t.age > 0]
            if cfg.max_tracks and len(T) >= cfg.max_tracks and lost:
                i = min(lost, key=lambda k: (np.hypot(T[k].last_obs_xy[0] - d["centroid"][0], T[k].last_obs_xy[1] - d["centroid"][1]), T[k].tid))
                matched_t.add(i)
                t = T[i]
                self._reset_to(t, d, step)
                out_obs.append({"id": t.tid, "det": j, "new": False, "uncertain": True, "reason": "two-egg prior relink (nearest lost track, any distance)"})
                continue
            b = d["bbox"]
            w, h = b[2] - b[0], b[3] - b[1]
            dm = 0.5 * (w + h)
            t = KTrack(self._next, np.array([[d["centroid"][0], 0.0], [d["centroid"][1], 0.0]]), np.diag([(cfg.sigma_meas * dm) ** 2, dm ** 2]), w, h, step, step,
                       last_obs_xy=(d["centroid"][0], d["centroid"][1]), obs_kinds=["complete"])
            self._next += 1
            T.append(t)
            out_obs.append({"id": t.tid, "det": j, "new": True, "uncertain": False, "reason": ""})
        for t in T:
            if 0 < t.age <= cfg.max_predicted_display:
                cx, cy = t.center()
                out_pred.append({"id": t.tid, "bbox": [cx - t.w / 2, cy - t.h / 2, cx + t.w / 2, cy + t.h / 2], "age": t.age})
        self.closed += [t for t in T if t.age > cfg.max_age]
        self.tracks = [t for t in T if t.age <= cfg.max_age]
        return sorted(out_obs, key=lambda o: (o["id"] is None, o["id"] or 0, o["det"])), sorted(out_pred, key=lambda o: o["id"])

    def all_tracks(self) -> list[KTrack]:
        return sorted(self.closed + self.tracks, key=lambda t: t.tid)
