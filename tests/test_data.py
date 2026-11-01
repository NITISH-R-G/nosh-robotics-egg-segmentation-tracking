import cv2
import numpy as np
import pytest

from src.data.inventory import image_id_from_path, inventory


def _write(p, arr):
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), arr)


def test_inventory_reads_images_and_ids(tmp_path):
    _write(tmp_path / "train/egg/not_damaged_1_jpg.rf.abc123.jpg", np.zeros((8, 6, 3), np.uint8))
    df = inventory(tmp_path)
    assert len(df) == 1
    r = df.iloc[0]
    assert (r.height, r.width, r.channels) == (8, 6, 3)
    assert r.image_id == "not_damaged_1" and r.original_split == "train" and r.label_folder == "egg"
    assert len(r.sha256) == 64 and r.readable


def test_corrupted_image_is_flagged_not_dropped(tmp_path):
    p = tmp_path / "train/egg/bad_jpg.rf.aa.jpg"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"not a jpeg")
    df = inventory(tmp_path)
    assert len(df) == 1 and not df.iloc[0].readable


def test_empty_and_missing_dataset(tmp_path):
    with pytest.raises(ValueError):
        inventory(tmp_path)
    with pytest.raises(FileNotFoundError):
        inventory(tmp_path / "nope")


def test_unexpected_extensions_ignored(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    _write(tmp_path / "b.png", np.zeros((4, 4, 3), np.uint8))
    df = inventory(tmp_path)
    assert list(df.path) == ["b.png"]


def test_image_id_fallback():
    assert image_id_from_path("foo/bar.jpg") == "bar"
    assert image_id_from_path("x/not_damaged_9_jpg.rf.ffee.jpg") == "not_damaged_9"
