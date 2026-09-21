"""
Movement heatmap generation (spec Module 2: "2D Gaussian Kernel Heatmap
Accumulators"; SAMGRAHA proposal mockup: the staples-bay dwell heatmap).

A coarse grid accumulates track ground-points every frame, with exponential
decay so the map reflects recent activity rather than an ever-growing total.
Two layers are kept: `traffic` (every active track) and `dwell` (only
near-stationary tracks), matching the spec's "Traffic Flow Heatmap" vs.
"Attention/Dwell Heatmap" distinction. Only this aggregated grid is ever
kept in memory -- raw per-frame trajectory coordinates are discarded the
same frame they're produced, consistent with the platform's zero-PII design.
"""
import threading
import time

import cv2
import numpy as np


class HeatmapAccumulator:
    def __init__(self, width: int, height: int, cell_size: int = 10,
                 decay: float = 0.995, dwell_speed_threshold_px: float = 2.0):
        self.cell_size = max(1, cell_size)
        self.cols = max(1, width // self.cell_size)
        self.rows = max(1, height // self.cell_size)
        self.decay = decay
        self.dwell_speed_threshold_px = dwell_speed_threshold_px
        self._lock = threading.Lock()
        self.traffic = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.dwell = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.last_update = time.time()

    def update(self, tracks):
        with self._lock:
            self.traffic *= self.decay
            self.dwell *= self.decay
            for t in tracks:
                gx = int(t.centroid[0] // self.cell_size)
                gy = int(t.centroid[1] // self.cell_size)
                if not (0 <= gx < self.cols and 0 <= gy < self.rows):
                    continue
                self.traffic[gy, gx] += 1.0
                if t.prev_centroid is not None:
                    dx = t.centroid[0] - t.prev_centroid[0]
                    dy = t.centroid[1] - t.prev_centroid[1]
                    if (dx * dx + dy * dy) ** 0.5 < self.dwell_speed_threshold_px:
                        self.dwell[gy, gx] += 1.0
            self.last_update = time.time()

    def render_png_bytes(self, layer: str = "traffic", out_size=None) -> bytes:
        with self._lock:
            grid = self.traffic if layer == "traffic" else self.dwell
            grid = grid.copy()
        peak = float(grid.max())
        norm = (grid / peak * 255.0).astype(np.uint8) if peak > 0 else grid.astype(np.uint8)
        if out_size:
            norm = cv2.resize(norm, out_size, interpolation=cv2.INTER_LINEAR)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        ok, buf = cv2.imencode(".png", colored)
        return buf.tobytes() if ok else b""
