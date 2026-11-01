import numpy as np

from src.tracking.detect import VideoDetectorConfig, detect
from src.tracking.sampling import EXPERIMENT_END, EXPERIMENT_START, is_dev, sample_frames, split_frames
from src.tracking.tracker import Tracker, TrackerConfig


def det(cx, cy, r=30):
    return {"bbox": [cx - r, cy - r, cx + r, cy + r], "centroid": [cx, cy]}


def run(seq, cfg=None):
    tr = Tracker(cfg or TrackerConfig())
    return [tr.update(i, d) for i, d in enumerate(seq)], tr


def ids(out):
    return [sorted(o["id"] for o in step) for step in out]


def test_stable_ids_for_two_slowly_moving_eggs():
    seq = [[det(100 + 2 * i, 200), det(400 - 2 * i, 220)] for i in range(20)]
    out, tr = run(seq)
    assert set(map(tuple, ids(out))) == {(1, 2)}
    assert len(tr.all_tracks()) == 2


def test_missing_detection_keeps_identity():
    seq = [[det(100, 200), det(400, 220)] for _ in range(5)] + [[det(400, 220)]] * 3 + [[det(100, 200), det(400, 220)]] * 3
    out, tr = run(seq)
    assert ids(out)[-1] == [1, 2]
    assert len(tr.all_tracks()) == 2


def test_temporary_occlusion_relinks_to_same_id_when_object_moved_within_growing_gate():
    seq = [[det(100, 200)]] * 3 + [[]] * 4 + [[det(160, 200)]]
    out, tr = run(seq)
    assert ids(out)[-1] == [1]


def test_loss_longer_than_max_age_creates_new_id_and_is_reported_as_two_tracks():
    cfg = TrackerConfig(max_age=3)
    seq = [[det(100, 200)]] * 2 + [[]] * 6 + [[det(100, 200)]]
    out, tr = run(seq, cfg)
    assert ids(out)[-1] == [2]
    assert [t.tid for t in tr.all_tracks()] == [1, 2]


def test_two_eggs_crossing_paths_do_not_swap_when_far_apart_per_step():
    seq = [[det(100 + 10 * i, 200), det(300 - 10 * i, 260)] for i in range(10)]
    out, _ = run(seq)
    a = [o["id"] for step in out for o in step if o["det"] == 0]
    assert set(a) == {1}


def test_new_detection_far_from_all_tracks_gets_new_id():
    out, tr = run([[det(100, 100)], [det(100, 100), det(800, 500)]])
    assert ids(out)[-1] == [1, 2]


def test_deterministic_output():
    seq = [[det(100 + i, 200), det(400, 220 + i)] for i in range(15)]
    a, _ = run(seq)
    b, _ = run(seq)
    assert a == b


def test_detector_finds_white_ellipse_and_ignores_textured_and_dark():
    img = np.full((540, 960, 3), (40, 50, 60), np.uint8)
    import cv2
    cv2.ellipse(img, (300, 300), (45, 35), 0, 0, 360, (250, 250, 250), -1)
    # high-frequency white/dark checker (shirt-like): fails the solidity/fill filters
    for y in range(40, 160, 8):
        for x in range(500, 800, 8):
            if (x // 8 + y // 8) % 2 == 0:
                img[y:y + 8, x:x + 8] = 250
    d = detect(img, VideoDetectorConfig())
    assert len(d) == 1
    cx, cy = d[0]["centroid"]
    assert abs(cx - 300) < 5 and abs(cy - 300) < 5


def test_detector_scales_boxes_to_original_resolution():
    import cv2
    img = np.full((1080, 1920, 3), (40, 50, 60), np.uint8)
    cv2.ellipse(img, (600, 600), (90, 70), 0, 0, 360, (250, 250, 250), -1)
    d = detect(img, VideoDetectorConfig())
    assert len(d) == 1
    x1, y1, x2, y2 = d[0]["bbox"]
    assert 480 < x1 < 520 and 680 < x2 < 720


def test_sampling_is_deterministic_and_dev_heldout_are_disjoint_and_cover():
    fr = sample_frames()
    assert fr == sample_frames() and fr[0] == EXPERIMENT_START and fr[-1] < EXPERIMENT_END
    dev, held = split_frames(fr)
    assert not set(dev) & set(held) and sorted(dev + held) == fr
    assert all(is_dev(f) for f in dev) and not any(is_dev(f) for f in held)


def test_two_egg_prior_relinks_lost_track_at_any_distance_and_default_does_not():
    seq = [[det(100, 200), det(400, 220)]] * 3 + [[det(400, 220)]] * 3 + [[det(1500, 300), det(400, 220)]]
    out, tr = run(seq, TrackerConfig(max_tracks=2))
    assert ids(out)[-1] == [1, 2] and len(tr.all_tracks()) == 2
    out0, tr0 = run(seq, TrackerConfig())
    assert ids(out0)[-1] == [2, 3]


def test_evaluate_counts_misses_switches_and_gaps(tmp_path):
    from src.tracking.evaluate import evaluate, label_detections, load_segments
    segs = load_segments()
    assert label_detections(300, [500.0, 150.0], segs) == ["P", "Q"]   # L=Q;R=P by x order
    assert label_detections(600, [500.0, 1300.0], segs) == ["P", "Q"]  # swapped order segment
    assert label_detections(520, [480.0], segs) == ["P"]
    rows = []
    frames = [300, 306, 312]
    for f, i1, i2 in [(300, 1, 2), (306, 1, 2), (312, 2, 1)]:   # ids swap at 312
        rows.append({"frame": f, "id": i1, "x1": 100, "x2": 200})
        rows.append({"frame": f, "id": i2, "x1": 500, "x2": 600})
    r = evaluate(rows, frames, segs, set())
    assert r["id_switches"] == 2 and r["coverage"] == 1.0 and r["missed"] == 0
    r2 = evaluate(rows[:-2], frames, segs, set())   # drop both detections at 312 -> 2 missed eggs
    assert r2["missed"] == 2 and r2["coverage"] == 4 / 6
