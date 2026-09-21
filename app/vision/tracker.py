"""
Person detection + kinematic multi-object tracking (Section 3.1 / Module 2).

Two interchangeable backends are provided behind the same `PersonTracker`
interface:

  - MotionBlobTracker (default): background subtraction + centroid-distance
    association. No ML weights or GPU required, so the platform can be
    demoed on any laptop immediately. This is a stand-in for the spec's
    "Shopper Flow: YOLOv8n/YOLOv10n + ByteTrack" pipeline.

  - YoloByteTrackTracker (optional, `--detector yolo`): the real detector
    described in the spec, using Ultralytics YOLOv8n person-class detection
    with its built-in ByteTrack integration. Requires `pip install
    ultralytics torch` (not installed by default -- see requirements.txt).

Per the Zero-PII design (Section 6 / Module 4.3): both backends emit only
bounding-box centroids and ephemeral integer track IDs. No appearance
embeddings, face detection, or Re-ID features are computed anywhere here,
and tracks older than `max_track_age_seconds` are permanently dropped.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.geometry import euclidean_distance


@dataclass
class Track:
    track_id: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    first_seen: float
    last_seen: float
    prev_centroid: Tuple[float, float] = None

    @property
    def dwell_seconds(self) -> float:
        return self.last_seen - self.first_seen


class PersonTracker:
    def update(self, frame) -> List[Track]:
        raise NotImplementedError


class MotionBlobTracker(PersonTracker):
    def __init__(self, min_area: int = 900, max_age_seconds: float = 2.0,
                 max_match_distance: float = 80.0):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=True
        )
        self.min_area = min_area
        self.max_age_seconds = max_age_seconds
        self.max_match_distance = max_match_distance
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    def _detect_centroids(self, frame) -> List[Tuple[Tuple[float, float], Tuple[int, int, int, int]]]:
        mask = self._bg.apply(frame)
        # Drop shadow pixels (value 127 in MOG2's shadow-detection output).
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            centroid = (x + w / 2.0, y + h)  # ground-contact point (feet), not head
            detections.append((centroid, (x, y, w, h)))
        return detections

    def update(self, frame) -> List[Track]:
        now = time.time()
        detections = self._detect_centroids(frame)
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self._tracks.keys())

        pairs = []
        for tid in unmatched_tracks:
            for di in unmatched_detections:
                d = euclidean_distance(self._tracks[tid].centroid, detections[di][0])
                pairs.append((d, tid, di))
        pairs.sort(key=lambda p: p[0])

        matched_tracks, matched_dets = set(), set()
        for dist, tid, di in pairs:
            if tid in matched_tracks or di in matched_dets:
                continue
            if dist > self.max_match_distance:
                continue
            centroid, bbox = detections[di]
            track = self._tracks[tid]
            track.prev_centroid = track.centroid
            track.centroid = centroid
            track.bbox = bbox
            track.last_seen = now
            matched_tracks.add(tid)
            matched_dets.add(di)

        for di, (centroid, bbox) in enumerate(detections):
            if di in matched_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(
                track_id=tid, centroid=centroid, bbox=bbox,
                first_seen=now, last_seen=now, prev_centroid=centroid,
            )

        stale = [tid for tid, t in self._tracks.items() if now - t.last_seen > self.max_age_seconds]
        for tid in stale:
            del self._tracks[tid]

        return list(self._tracks.values())


class YoloByteTrackTracker(PersonTracker):
    """Production backend matching the spec's YOLOv8n + ByteTrack pipeline."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5,
                 max_age_seconds: float = 2.0):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO backend requires `pip install ultralytics torch`. "
                "Fall back to --detector motion if you don't need GPU-grade accuracy."
            ) from exc
        self._model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.max_age_seconds = max_age_seconds
        self._last_seen: Dict[int, float] = {}
        self._first_seen: Dict[int, float] = {}
        self._prev_centroid: Dict[int, Tuple[float, float]] = {}

    def update(self, frame) -> List[Track]:
        now = time.time()
        results = self._model.track(
            frame, persist=True, classes=[0], conf=self.confidence_threshold,
            tracker="bytetrack.yaml", verbose=False,
        )
        tracks: List[Track] = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.int().tolist()
            xyxy = boxes.xyxy.tolist()
            for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                centroid = ((x1 + x2) / 2.0, y2)  # ground-contact point
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                if tid not in self._first_seen:
                    self._first_seen[tid] = now
                prev = self._prev_centroid.get(tid, centroid)
                tracks.append(Track(
                    track_id=tid, centroid=centroid, bbox=bbox,
                    first_seen=self._first_seen[tid], last_seen=now, prev_centroid=prev,
                ))
                self._prev_centroid[tid] = centroid
                self._last_seen[tid] = now

        stale = [tid for tid, t in self._last_seen.items() if now - t > self.max_age_seconds]
        for tid in stale:
            self._last_seen.pop(tid, None)
            self._first_seen.pop(tid, None)
            self._prev_centroid.pop(tid, None)

        return tracks


def build_tracker(backend: str, config) -> PersonTracker:
    if backend == "yolo":
        return YoloByteTrackTracker(
            model_path=config.tracker.yolo_model_path,
            confidence_threshold=config.tracker.confidence_threshold,
            max_age_seconds=config.tracker.max_track_age_seconds,
        )
    return MotionBlobTracker(
        min_area=config.tracker.min_track_area,
        max_age_seconds=config.tracker.max_track_age_seconds,
    )
