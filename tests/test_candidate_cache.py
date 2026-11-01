import numpy as np

from src.segmentation.cache_candidates import candidate_record, pack_masks, unpack_masks


def _masks():
    a = np.zeros((13, 17), bool); a[2:7, 3:9] = True
    b = np.zeros((13, 17), bool); b[0:4, 10:17] = True
    return [a, b]


def test_pack_unpack_roundtrip_non_multiple_of_8(tmp_path):
    ms = _masks()
    packed, shape = pack_masks(ms)
    p = tmp_path / "x.npz"
    np.savez_compressed(p, masks=packed, shape=np.array(shape))
    out = unpack_masks(p)
    assert out.shape == (2, 13, 17) and (out == np.stack(ms)).all()


def test_candidate_record_fields():
    m = _masks()[1]
    c = {"bbox_xywh": [10, 0, 7, 4], "predicted_iou": 0.9, "stability_score": 0.8, "point_coords": [12.0, 2.0],
         "crop_box": [0, 0, 17, 13]}
    r = candidate_record("img", 3, "masks/img.npz", m, c)
    assert r["candidate_id"] == "img#3" and r["area"] == int(m.sum()) and r["touches_border"]
    assert abs(r["centroid_xy"][0] - 13.0) < 1e-9 and abs(r["centroid_xy"][1] - 1.5) < 1e-9
