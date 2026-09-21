"""
Planar projective geometry: pixel -> metric (cm) coordinate mapping
(SAMGRAHA Research Foundations: "Planar Projective Geometry (x' = Hx) ...
for metric shelf-depth estimation and floor ground-plane dwell mapping").

Two ways to obtain a PlanarHomography:
  - Static calibration: four known pixel<->cm point correspondences typed
    once into store_layout.json (e.g. the shelf rail corners measured with
    a tape).
  - ArucoHomographyCalibrator: four ArUco markers placed at known cm
    positions on the shelf upright/floor are re-detected on demand, so a
    bumped/drifted camera "auto-heals" its calibration instead of silently
    reporting wrong metric gaps (Feasibility & Viability: "Camera Drift ->
    ArUco/AprilTag Homography Auto-Healing").
"""
import time

import cv2
import numpy as np


class PlanarHomography:
    def __init__(self, pixel_points, metric_points):
        """pixel_points / metric_points: 4 corresponding (x, y) point pairs
        (metric in centimeters) on the same physical plane."""
        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(metric_points, dtype=np.float32)
        matrix, _ = cv2.findHomography(src, dst)
        if matrix is None:
            raise ValueError("could not compute a homography from the given point correspondences")
        self.matrix = matrix
        self.calibrated_at = time.time()

    def project_point(self, point):
        px = np.array([[point]], dtype=np.float32)
        out = cv2.perspectiveTransform(px, self.matrix)
        return float(out[0][0][0]), float(out[0][0][1])

    def project_distance_cm(self, p1, p2) -> float:
        """Metric (cm) straight-line distance between two pixel points on
        the calibrated plane -- used for shelf gap widths and floor
        distances alike."""
        x1, y1 = self.project_point(p1)
        x2, y2 = self.project_point(p2)
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


class ArucoHomographyCalibrator:
    def __init__(self, marker_ids_to_metric_cm: dict, dictionary=cv2.aruco.DICT_4X4_50):
        """marker_ids_to_metric_cm: {aruco_marker_id: (x_cm, y_cm)} for the
        four (or more) markers fixed to the shelf/floor at known positions."""
        if len(marker_ids_to_metric_cm) < 4:
            raise ValueError("need at least 4 ArUco markers to compute a homography")
        self.marker_ids_to_metric_cm = marker_ids_to_metric_cm
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, cv2.aruco.DetectorParameters())
        self._last_homography: PlanarHomography = None

    def try_recalibrate(self, frame, max_age_seconds: float = 30.0):
        """Re-detects the configured markers in `frame` and recomputes the
        homography if all of them are visible this frame. If some are
        occluded, falls back to the last known-good calibration as long as
        it's not older than max_age_seconds; returns None if neither is
        available (caller should skip metric conversion for this frame)."""
        corners, ids, _ = self._detector.detectMarkers(frame)
        if ids is not None:
            detected = {int(marker_id[0]): corner[0].mean(axis=0) for marker_id, corner in zip(ids, corners)}
            required = set(self.marker_ids_to_metric_cm.keys())
            if required.issubset(detected.keys()):
                pixel_points = [tuple(detected[mid]) for mid in self.marker_ids_to_metric_cm]
                metric_points = [self.marker_ids_to_metric_cm[mid] for mid in self.marker_ids_to_metric_cm]
                self._last_homography = PlanarHomography(pixel_points, metric_points)
                return self._last_homography

        if self._last_homography and (time.time() - self._last_homography.calibrated_at) <= max_age_seconds:
            return self._last_homography
        return None
