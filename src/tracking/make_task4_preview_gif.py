"""Make a short preview GIF for the README from the T3b footage-overlay video (NOT from either schematic video).

Source: outputs/video/final/egg_tracking_T3b_footage_overlay_LOCAL_ONLY_v2.mp4 (produced by src/tracking/render_t3b_overlay.py from the saved T3b tracks.csv; the same file is on Google Drive).
The preview shows source frames 498-617 (hand lifts one egg, predicted / fragment / uncertain overlays, re-acquisition), every 3rd video frame (8 fps), a global 112-colour palette seeded with the overlay colours.
Usage: python -m src.tracking.make_task4_preview_gif [out.gif] [width]
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import cv2
from PIL import Image

SRC = "outputs/video/final/egg_tracking_T3b_footage_overlay_LOCAL_ONLY_v2.mp4"
DEFAULT_OUT = "outputs/figures/task4_t3b_overlay_preview.gif"
FIRST_SRC_FRAME, LAST_SRC_FRAME, STEP = 498, 618, 3      # source frames 498..617 every 3rd frame (8 fps); the video starts at source frame 240
VIDEO_START = 240


def main(out: str = DEFAULT_OUT, width: int = 640) -> None:
    cap = cv2.VideoCapture(SRC)
    h = int(round(width * 720 / 1280))
    frames = []
    for sf in range(FIRST_SRC_FRAME, LAST_SRC_FRAME, STEP):
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf - VIDEO_START)
        ok, f = cap.read()
        if not ok:
            raise SystemExit(f"cannot read video frame for source frame {sf}")
        frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))
    mosaic = Image.new("RGB", (width, h * 4 + 40))
    for i, k in enumerate((0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1)):
        mosaic.paste(frames[k], (0, h * i))
    swatch = [(60, 200, 60), (255, 165, 0), (255, 255, 0), (255, 40, 40), (255, 255, 255), (170, 170, 170), (0, 0, 0)]      # the exact overlay colours (ID 1, ID 2, predicted text, uncertain text, white, grey, black)
    for i, c in enumerate(swatch):
        mosaic.paste(Image.new("RGB", (width // len(swatch) + 1, 40), c), (i * (width // len(swatch)), h * 4))
    pal = mosaic.quantize(colors=112, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [f.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{out}.{os.getpid()}.tmp")                # write privately, then replace atomically so a concurrent run never leaves a half-written GIF
    try:
        q[0].save(tmp, format="GIF", save_all=True, append_images=q[1:], duration=int(1000 / 8), loop=0, optimize=True, disposal=1)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    print(out, Path(out).stat().st_size, "bytes;", len(q), "frames;", f"{width}x{h}", "; source sha256", hashlib.sha256(Path(SRC).read_bytes()).hexdigest())


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []), *([int(sys.argv[2])] if len(sys.argv) > 2 else []))
