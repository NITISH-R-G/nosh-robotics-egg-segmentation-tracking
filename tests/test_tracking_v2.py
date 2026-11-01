import cv2
import numpy as np

from src.tracking.detect import VideoDetectorConfig, detect
from src.tracking.detect_v2 import PartialConfig, detect_with_partials
from src.tracking.tracker_v2 import KalmanTracker, KConfig


def det(cx, cy, r=60, kind="complete"):
    return {"bbox": [cx - r, cy - r, cx + r, cy + r], "centroid": [float(cx), float(cy)], "kind": kind}


def run(seq, cfg=None):
    tr = KalmanTracker(cfg or KConfig())
    obs, pred = [], []
    for i, d in enumerate(seq):
        o, p = tr.update(i, d)
        obs.append(o)
        pred.append(p)
    return obs, pred, tr


def ids(o):
    return sorted(x["id"] for x in o if x["id"] is not None)


def test_temporary_disappearance_predicts_then_same_id_returns():
    seq = [[det(100 + 10 * i, 300)] for i in range(5)] + [[]] * 3 + [[det(100 + 10 * 8, 300)]]
    obs, pred, tr = run(seq)
    assert ids(obs[-1]) == [1]
    assert all(len(p) == 1 and p[0]["id"] == 1 for p in pred[5:8])     # predicted boxes reported while missing
    assert len(tr.all_tracks()) == 1


def test_predicted_boxes_are_not_observations():
    seq = [[det(100, 300)]] * 4 + [[]] * 2
    obs, pred, _ = run(seq)
    assert obs[-1] == [] and len(pred[-1]) == 1
    assert pred[-1][0]["age"] == 2 and "det" not in pred[-1][0]


def test_prediction_follows_velocity_and_is_damped():
    seq = [[det(100 + 20 * i, 300)] for i in range(8)] + [[]] * 6
    _, pred, _ = run(seq, KConfig(vel_damp=0.5))
    xs = [(p[0]["bbox"][0] + p[0]["bbox"][2]) / 2 for p in pred[8:]]
    assert xs[0] > 100 + 20 * 7 and all(b - a < 20 for a, b in zip(xs, xs[1:]))   # continues forward, decelerating
    assert xs[-1] - xs[0] < 40                                                  # no runaway drift


def test_long_disappearance_expires_and_new_id_after_return_without_prior():
    cfg = KConfig(max_age=4, max_tracks=0)
    seq = [[det(100, 300)]] * 3 + [[]] * 8 + [[det(100, 300)]]
    obs, _, tr = run(seq, cfg)
    assert ids(obs[-1]) == [2]
    assert [t.tid for t in tr.all_tracks()] == [1, 2]


def test_two_objects_close_together_keep_ids():
    seq = [[det(300 + 2 * i, 300), det(480 - 2 * i, 310)] for i in range(12)]
    obs, _, tr = run(seq)
    assert all(ids(o) == [1, 2] for o in obs) and len(tr.all_tracks()) == 2
    first = {o["det"]: o["id"] for o in obs[0]}
    assert {o["det"]: o["id"] for o in obs[-1]} == first


def test_two_objects_exchanging_relative_positions_keep_identity_by_motion():
    n = 14
    seq = [[det(200 + 40 * i, 300), det(740 - 40 * i, 300 + 3 * i)] for i in range(n)]   # cross at the middle
    obs, _, _ = run(seq)
    a = [o["id"] for step in obs for o in step if o["det"] in (0, 1) and abs(seq[obs.index(step)][o["det"]]["centroid"][0] - (200 + 40 * obs.index(step))) < 1]
    assert set(a) == {1}                        # the object that started on the left is always id 1


def test_velocity_prediction_beats_nearest_neighbour_when_they_disagree():
    # id1 moves right at 50 px/step; a detection 20 px behind its last position is nearer than the one 50 px ahead,
    # but the constant-velocity prediction says the one ahead is the same object (the nearest-neighbour T1 tracker gets this wrong)
    seq = [[det(100 + 50 * i, 300)] for i in range(6)] + [[det(330, 300), det(400, 300)]]
    obs, _, _ = run(seq, KConfig(max_tracks=0))
    last = {o["det"]: o["id"] for o in obs[-1]}
    assert last[1] == 1 and last[0] == 2


def test_ambiguous_association_is_flagged_uncertain():
    # two tracks, one detection equidistant from both -> assigned but flagged
    seq = [[det(200, 300), det(400, 300)]] * 3 + [[det(300, 300)]]
    obs, _, _ = run(seq)
    last = [o for o in obs[-1] if o["id"] is not None]
    assert len(last) == 1 and last[0]["uncertain"] is True


def test_two_egg_prior_relink_is_flagged_uncertain_and_keeps_two_ids():
    cfg = KConfig(max_age=30)
    seq = [[det(100, 300), det(500, 300)]] * 3 + [[det(500, 300)]] * 4 + [[det(1500, 600), det(500, 300)]]
    obs, _, tr = run(seq, cfg)
    far = [o for o in obs[-1] if o["det"] == 0 or o["id"] == 1]
    assert ids(obs[-1]) == [1, 2] and any(o["uncertain"] and "prior" in o["reason"] for o in obs[-1])
    assert len(tr.all_tracks()) == 2


def test_partial_detection_updates_a_track_but_never_creates_one():
    seq = [[det(100, 300)]] * 3 + [[det(100, 300, r=25, kind="partial"), det(900, 700, r=25, kind="partial")]]
    obs, _, tr = run(seq)
    kinds = {o["det"]: o["id"] for o in obs[-1]}
    assert kinds[0] == 1 and kinds[1] is None and len(tr.all_tracks()) == 1


def test_partial_has_less_pull_than_complete():
    cfg = KConfig()
    tr1 = KalmanTracker(cfg); tr2 = KalmanTracker(cfg)
    for tr in (tr1, tr2):
        for i in range(3):
            tr.update(i, [det(100, 300)])
    tr1.update(3, [det(160, 300, kind="complete")])
    tr2.update(3, [det(160, 300, kind="partial")])
    assert tr2.tracks[0].center()[0] < tr1.tracks[0].center()[0]


def test_deterministic_output():
    seq = [[det(100 + 7 * i, 300), det(600 - 5 * i, 320)] if i % 5 else [det(100 + 7 * i, 300)] for i in range(20)]
    a, pa, _ = run(seq)
    b, pb, _ = run(seq)
    assert a == b and pa == pb


def test_stable_ids_when_detections_return_after_gap():
    seq = [[det(100, 300), det(500, 300)]] * 3 + [[det(500, 300)]] * 2 + [[det(100, 300), det(500, 300)]] * 3
    obs, _, tr = run(seq)
    assert ids(obs[-1]) == [1, 2] and len(tr.all_tracks()) == 2


def test_detector_complete_detections_are_identical_to_strict_detector():
    img = np.full((540, 960, 3), (40, 50, 60), np.uint8)
    cv2.ellipse(img, (300, 300), (45, 35), 0, 0, 360, (250, 250, 250), -1)
    base = VideoDetectorConfig()
    strict = detect(img, base)
    v2 = [d for d in detect_with_partials(img, base, PartialConfig()) if d["kind"] == "complete"]
    assert [d["bbox"] for d in strict] == [d["bbox"] for d in v2]


def test_detector_finds_partial_egg_fragment_split_by_a_finger_and_disabled_flag_removes_it():
    img = np.full((540, 960, 3), (40, 50, 60), np.uint8)
    cv2.ellipse(img, (300, 300), (60, 45), 0, 0, 360, (250, 250, 250), -1)
    cv2.rectangle(img, (310, 240), (345, 360), (60, 90, 150), -1)      # skin-coloured finger covering the right part
    base = VideoDetectorConfig()
    d = detect_with_partials(img, base, PartialConfig())
    assert not any(x["kind"] == "complete" for x in d)               # strict filters reject the cut egg
    assert any(x["kind"] == "partial" for x in d)
    assert detect_with_partials(img, base, PartialConfig(enabled=False)) == []


def test_tiny_white_speck_is_not_a_partial_detection():
    img = np.full((540, 960, 3), (40, 50, 60), np.uint8)
    cv2.circle(img, (200, 200), 6, (250, 250, 250), -1)
    assert detect_with_partials(img, VideoDetectorConfig(), PartialConfig()) == []
