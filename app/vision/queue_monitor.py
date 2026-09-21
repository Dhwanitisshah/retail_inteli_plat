"""
Predictive Queue Intelligence (Section 3.2 + Section 4.1 Core Queue Alerting
Heuristic). This mirrors the spec's reference pseudocode:

    qualifying_shoppers = count of tracks inside ROI_queue with
                           dwell_seconds >= DWELL_TIME_SECONDS
    if qualifying_shoppers >= QUEUE_THRESHOLD:
        if continuous_congestion_time >= CONFIRMATION_WINDOW:
            dispatch_alert(...)

Wait time uses the spec's simplified formula (Section 3.2):
    Wait Time ~= (People in Queue * Avg Service Time per Customer) / Active Cashiers
"""
import time
from dataclasses import dataclass
from typing import List

from app.geometry import Polygon
from app.vision.tracker import Track
from config import QueueConfig


@dataclass
class QueueState:
    lane_id: str
    people_count: int
    qualifying_shoppers: int
    estimated_wait_seconds: float
    alert_triggered: bool
    continuous_congestion_seconds: float


class QueueMonitor:
    def __init__(self, lane_id: str, polygon: Polygon, config: QueueConfig,
                 active_cashiers: int = 1, avg_service_time_seconds: float = 90.0):
        self.lane_id = lane_id
        self.polygon = polygon
        self.config = config
        self.active_cashiers = active_cashiers
        self.avg_service_time_seconds = avg_service_time_seconds
        self._entered_at = {}          # track_id -> time entered ROI_queue
        self._congestion_since = None  # time qualifying_shoppers first breached threshold

    def evaluate(self, tracks: List[Track]) -> QueueState:
        now = time.time()
        in_polygon_ids = set()
        qualifying_shoppers = 0

        for track in tracks:
            if not self.polygon.contains(track.centroid):
                continue
            in_polygon_ids.add(track.track_id)
            entered_at = self._entered_at.setdefault(track.track_id, now)
            dwell_seconds = now - entered_at
            if dwell_seconds >= self.config.dwell_time_seconds:
                qualifying_shoppers += 1

        for track_id in list(self._entered_at.keys()):
            if track_id not in in_polygon_ids:
                del self._entered_at[track_id]

        people_count = len(in_polygon_ids)
        estimated_wait_seconds = (
            people_count * self.avg_service_time_seconds
        ) / max(self.active_cashiers, 1)

        if qualifying_shoppers >= self.config.threshold:
            if self._congestion_since is None:
                self._congestion_since = now
            continuous_congestion_seconds = now - self._congestion_since
            alert_triggered = continuous_congestion_seconds >= self.config.confirmation_window
        else:
            self._congestion_since = None
            continuous_congestion_seconds = 0.0
            alert_triggered = False

        return QueueState(
            lane_id=self.lane_id,
            people_count=people_count,
            qualifying_shoppers=qualifying_shoppers,
            estimated_wait_seconds=estimated_wait_seconds,
            alert_triggered=alert_triggered,
            continuous_congestion_seconds=continuous_congestion_seconds,
        )
