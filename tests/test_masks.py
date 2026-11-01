import numpy as np
import pytest

from src.segmentation.masks import load_mask_png, save_mask_png, to_binary


def test_to_binary_accepts_bool_01_255():
    for m in (np.eye(3, dtype=bool), np.eye(3, dtype=np.uint8), np.eye(3, dtype=np.uint8) * 255):
        b = to_binary(m)
        assert b.dtype == np.uint8 and set(np.unique(b)) == {0, 1}


def test_to_binary_rejects_bad_input():
    with pytest.raises(ValueError):
        to_binary(np.full((3, 3), 7, np.uint8))
    with pytest.raises(TypeError):
        to_binary(np.zeros((3, 3), np.float32))
    with pytest.raises(ValueError):
        to_binary(np.zeros((3, 3, 3), np.uint8))
    with pytest.raises(ValueError):
        to_binary(np.zeros((3, 3), np.uint8), shape=(4, 4))


def test_png_roundtrip_preserves_shape_and_values(tmp_path):
    m = np.zeros((17, 23), np.uint8)
    m[3:9, 4:12] = 1
    p = tmp_path / "sub" / "m.png"
    save_mask_png(p, m)
    out = load_mask_png(p, shape=(17, 23))
    assert out.shape == m.shape and (out == m).all()


def test_load_corrupt_png(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"junk")
    with pytest.raises(IOError):
        load_mask_png(p)
