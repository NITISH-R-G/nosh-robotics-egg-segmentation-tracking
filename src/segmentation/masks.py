"""Mask normalisation, validation and PNG I/O.

In memory masks are uint8 arrays of {0, 1}. On disk they are single-channel 8-bit PNGs
with {0, 255} (so they display in any viewer) at the original image size.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def to_binary(mask: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Return uint8 {0,1} from a bool / {0,1} / {0,255} array; reject anything else.

    Never resizes: a shape mismatch is an error, not something to silently interpolate.
    """
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {m.shape}")
    if shape is not None and m.shape != tuple(shape):
        raise ValueError(f"mask shape {m.shape} != expected {tuple(shape)}")
    if m.dtype == bool:
        return m.astype(np.uint8)
    if not np.issubdtype(m.dtype, np.integer):
        raise TypeError(f"mask dtype must be bool or integer, got {m.dtype}")
    vals = set(np.unique(m).tolist())
    if vals <= {0, 1}:
        return m.astype(np.uint8)
    if vals <= {0, 255}:
        return (m > 0).astype(np.uint8)
    raise ValueError(f"mask has non-binary values: {sorted(vals)[:6]}")


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", to_binary(mask) * 255)
    if not ok:
        raise IOError(f"could not encode mask for {path}")
    buf.tofile(str(path))


def load_mask_png(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    m = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise IOError(f"could not decode mask: {path}")
    return to_binary(m, shape)
