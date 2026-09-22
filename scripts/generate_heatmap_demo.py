"""
Builds a demo heatmap overlay on a real store photo, using the platform's
actual HeatmapAccumulator (app/vision/heatmap.py) -- not a mockup. Synthetic
shopper positions stand in for what several minutes of real track data would
look like (clustered near active checkout lanes, thin along the walking
aisle, near-empty by the closed counter), then the same
`render_png_bytes()` the live dashboard calls is alpha-blended over the
source photo.

Usage:
    python scripts/generate_heatmap_demo.py
"""
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vision.heatmap import HeatmapAccumulator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PHOTO = os.path.join(ROOT, "assets", "demo_photos", "08_checkout_wide_aisle.png")
OUT_DIR = os.path.join(ROOT, "assets", "demo_outputs")

# Rough hotspot centers (px) picked from the source photo: busy tills,
# a thinner walking band across the aisle, and a near-empty patch by the
# closed counter -- (mean_x, mean_y, spread_px, relative_weight)
HOTSPOTS = [
    (260, 640, 70, 1.0),   # customers/staff clustered at an open till
    (520, 700, 60, 0.8),
    (760, 660, 55, 0.7),
    (980, 610, 50, 0.5),
    (700, 900, 260, 0.4),  # thin band along the main aisle walkway
    (1120, 640, 45, 0.08), # near the "COUNTER CLOSED" lane -- mostly quiet
]


def simulate_tracks(width, height, rng, n_people=14):
    """One synthetic 'frame' of person positions sampled from the hotspots."""
    weights = np.array([h[3] for h in HOTSPOTS], dtype=np.float64)
    weights /= weights.sum()
    tracks = []
    for i in range(n_people):
        cx, cy, spread, _ = HOTSPOTS[rng.choice(len(HOTSPOTS), p=weights)]
        x = float(np.clip(rng.normal(cx, spread), 0, width - 1))
        y = float(np.clip(rng.normal(cy, spread), 0, height - 1))
        tracks.append(SimpleNamespace(centroid=(x, y), prev_centroid=None))
    return tracks


def build_heatmap(width, height, frames=220, seed=7):
    rng = np.random.default_rng(seed)
    accumulator = HeatmapAccumulator(width=width, height=height, cell_size=14, decay=0.995)
    prev_positions = {}
    for _ in range(frames):
        tracks = simulate_tracks(width, height, rng)
        # Give dwelling clusters near-still frame-to-frame movement so the
        # dwell layer (not just traffic) picks up real signal too.
        for idx, t in enumerate(tracks):
            prev = prev_positions.get(idx)
            t.prev_centroid = prev if prev is not None and rng.random() < 0.7 else None
            prev_positions[idx] = t.centroid
        accumulator.update(tracks)
    return accumulator


def overlay_heatmap(photo_path, accumulator, max_alpha=0.75, noise_floor=0.06, gamma=0.6):
    """Per-pixel alpha blend, proportional to local density, instead of a
    flat cv2.addWeighted() -- JET's zero-value color is dark blue, not
    transparent, so a flat blend tints the whole floor. Zeroing anything
    below `noise_floor` and blending only in proportion to intensity keeps
    quiet areas showing the original photo untouched."""
    photo = cv2.imread(photo_path)
    if photo is None:
        raise FileNotFoundError(photo_path)
    h, w = photo.shape[:2]

    grid = accumulator.traffic.copy()
    peak = float(grid.max())
    norm = grid / peak if peak > 0 else grid
    norm = cv2.resize(norm, (w, h), interpolation=cv2.INTER_CUBIC)
    norm = np.clip(norm, 0.0, 1.0)
    norm[norm < noise_floor] = 0.0
    shaped = norm ** gamma  # boost mid-tones so moderate activity still reads clearly

    heat_color = cv2.applyColorMap((shaped * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = (shaped * max_alpha)[:, :, None]
    blended = (photo.astype(np.float32) * (1 - alpha) + heat_color.astype(np.float32) * alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    heat_only_png = accumulator.render_png_bytes("traffic", out_size=(w, h))
    heat_only = cv2.imdecode(np.frombuffer(heat_only_png, np.uint8), cv2.IMREAD_COLOR)
    return blended, heat_only


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    photo = cv2.imread(SRC_PHOTO)
    h, w = photo.shape[:2]

    accumulator = build_heatmap(w, h)
    overlay, heat_only = overlay_heatmap(SRC_PHOTO, accumulator)

    cv2.imwrite(os.path.join(OUT_DIR, "heatmap_overlay_checkout.png"), overlay)
    cv2.imwrite(os.path.join(OUT_DIR, "heatmap_raw_checkout.png"), heat_only)
    print(f"Wrote heatmap demo images to {OUT_DIR}")


if __name__ == "__main__":
    main()
