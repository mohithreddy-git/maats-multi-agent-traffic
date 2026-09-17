"""Generates a small library of genuinely-different synthetic demo clips
into data/traffic/, for evaluator video-switching demos.

The clips shipped earlier (backend/cv/configs/demo_videos/*.mp4) turned out
to be byte-identical placeholders, so switching between them doesn't visibly
change detection counts. These clips are deliberately different from each
other (vehicle count, and one with a different resolution/orientation) so a
live "switch video" demo actually shows a different result -- real synthetic
motion for MotionDetector's background-subtraction to find, not photographic
footage (see backend/cv/motion_detector.py's docstring for why).

Re-run this script any time to regenerate the library:
    python scripts/generate_demo_clips.py
"""
from __future__ import annotations

import os

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "traffic")


def _write_clip(path: str, box_count: int, size=(640, 480), frames: int = 90) -> None:
    # Boxes drift diagonally across the FULL frame (not just a bottom strip)
    # so they pass through every quadrant -- backend/cv/configs/demo_four_directions.json
    # assigns each direction only a quadrant-sized ROI (e.g. S = top half,
    # y<280), and a clip whose motion never reaches a given quadrant would
    # make that direction permanently show zero detections regardless of
    # which clip is picked there.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, size)
    width, height = size
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype="uint8")
        for lane in range(box_count):
            x = (40 + lane * 90 + i * 5) % (width - 60)
            y = (30 + lane * 70 + i * 3) % (height - 40)
            cv2.rectangle(frame, (x, y), (x + 50, y + 40), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    clips = {
        "empty.mp4": dict(box_count=0, size=(640, 480)),
        "light.mp4": dict(box_count=1, size=(640, 480)),
        "medium.mp4": dict(box_count=3, size=(640, 480)),
        "heavy.mp4": dict(box_count=6, size=(640, 480)),
        # different resolution + portrait orientation, to exercise "adapt
        # safely to a different resolution/orientation" without any ROI
        # recalibration -- proves the pipeline doesn't crash, not that the
        # default ROI is meaningful for it.
        "portrait_480x640.mp4": dict(box_count=2, size=(480, 640)),
    }
    for filename, kwargs in clips.items():
        path = os.path.join(OUT_DIR, filename)
        _write_clip(path, **kwargs)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
