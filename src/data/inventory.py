"""Dataset inventory: enumerate images, validate decodability, hash contents."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_STEM_RE = re.compile(r"^(?P<image_id>.+?)_jpg\.rf\.[0-9a-f]+$")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_id_from_path(path: Path) -> str:
    """Roboflow names files '<source>_jpg.rf.<hash>.jpg'; the source part is the stable id."""
    m = _STEM_RE.match(Path(path).stem)
    return m.group("image_id") if m else Path(path).stem


def inventory(root: Path) -> pd.DataFrame:
    """Return one row per image file under root.

    Columns: path (posix, relative to root), image_id, original_split, label_folder,
    sha256, height, width, channels, readable. Unreadable files are kept with readable=False.
    Raises FileNotFoundError if root is missing and ValueError if it contains no images.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        rows.append(
            {
                "path": rel.as_posix(),
                "image_id": image_id_from_path(p),
                "original_split": parts[0] if len(parts) > 1 else "",
                "label_folder": parts[1] if len(parts) > 2 else "",
                "sha256": sha256_file(p),
                "height": None if img is None else img.shape[0],
                "width": None if img is None else img.shape[1],
                "channels": None if img is None else img.shape[2],
                "readable": img is not None,
            }
        )
    if not rows:
        raise ValueError(f"no images with extensions {sorted(IMAGE_EXTS)} under {root}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/inventory.csv")
    a = ap.parse_args()
    df = inventory(Path(a.root))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} images, {int((~df.readable).sum())} unreadable -> {a.out}")
