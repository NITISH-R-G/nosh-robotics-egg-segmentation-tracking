"""Frame interval and development/held-out frame sets for the video (Task 4)."""
from __future__ import annotations

FPS = 24
EXPERIMENT_START = 240   # first frame with the table scene fully faded in (title card ends ~224, see frame_stats.csv)
EXPERIMENT_END = 1000    # exclusive; frame 1004 is the dark transition to the explanation card
STEP = 6                 # 4 frames/second sample


def sample_frames(start: int = EXPERIMENT_START, end: int = EXPERIMENT_END, step: int = STEP) -> list[int]:
    return list(range(start, end, step))


def is_dev(frame: int) -> bool:
    """Alternate 1-second blocks: even blocks are development, odd blocks are held out."""
    return (frame // FPS) % 2 == 0


def split_frames(frames: list[int]) -> tuple[list[int], list[int]]:
    return [f for f in frames if is_dev(f)], [f for f in frames if not is_dev(f)]
