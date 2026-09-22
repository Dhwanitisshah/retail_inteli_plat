"""
Generates short synthetic demo clips into data/demo_clips/ so the platform
can be exercised end-to-end without physical cameras. These are plain
OpenCV-drawn shapes, not real footage -- useful for wiring/logic demos, not
detection-accuracy demos (use a real webcam or recorded store footage with
`--detector yolo` for that).

Matches sample_config/store_layout.json's default geometry:
  - entrance_cam tripwire:  p1=(50,400) -> p2=(590,400)
  - checkout_cam queue ROI: [(20,20), (300,20), (300,460), (20,460)]
  - shelf_cam slot A4_T2_S1 rect: (0, 100, 210, 150)

Usage:
    python scripts/generate_demo_clips.py
"""
import os

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "demo_clips")
W, H = 640, 480


def generate_entrance_clip(path, n_frames=120, fps=20):
    """A person-blob walks top-to-bottom, crossing the y=400 tripwire once."""
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(n_frames):
        frame = np.full((H, W, 3), 40, dtype=np.uint8)
        t = i / n_frames
        cx, cy = 300, int(50 + t * 400)
        cv2.rectangle(frame, (cx - 25, cy - 50), (cx + 25, cy + 10), (220, 220, 220), -1)
        out.write(frame)
    out.release()


def generate_checkout_clip(path, n_frames=90, fps=15):
    """Several person-blobs sit inside the queue ROI, simulating a building line.
    Run with --loop and let it play for a couple of real-time minutes to see the
    dwell-time + confirmation-window alert actually fire (see README)."""
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    positions = [(60, 60), (60, 160), (60, 260), (60, 360)]
    for i in range(n_frames):
        frame = np.full((H, W, 3), 40, dtype=np.uint8)
        jitter = (i % 3) - 1
        for (x, y) in positions:
            cv2.rectangle(frame, (x + jitter, y), (x + 50 + jitter, y + 80), (220, 220, 220), -1)
        out.write(frame)
    out.release()


def generate_shelf_clip(path, n_frames=90, fps=15):
    """Slot A4_T2_S1 goes stocked -> empty -> restocked (each a third of the
    clip). Run with `--loop` for a couple of real minutes (shelf evaluation
    is throttled to every `evaluation_interval_seconds`, so it takes real
    wall-clock time, not video duration, to accumulate the 3 consecutive
    over-threshold reads the alert needs) to watch a restock task actually
    get created *and* resolved once the camera sees the shelf filled again."""
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    rng = np.random.default_rng(7)
    third = n_frames // 3
    for i in range(n_frames):
        frame = np.full((H, W, 3), 60, dtype=np.uint8)
        if i < third or i >= 2 * third:
            frame[100:250, 0:210] = rng.integers(0, 255, size=(150, 210, 3), dtype=np.uint8)
        out.write(frame)
    out.release()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    generate_entrance_clip(os.path.join(OUT_DIR, "entrance.mp4"))
    generate_checkout_clip(os.path.join(OUT_DIR, "checkout.mp4"))
    generate_shelf_clip(os.path.join(OUT_DIR, "shelf.mp4"))
    print(f"Wrote demo clips to {OUT_DIR}")
