"""
Predictive Queue Intelligence: dwell-threshold alerting (Section 3.2 +
Section 4.1 Core Queue Alerting Heuristic) plus a Congestion Index forecast
engine (SAMGRAHA Technical Approach: "Queue & Flow Engine ... calculates
Congestion Index (CI = lambda / (c * mu))").

Dwell-threshold state machine mirrors the spec's reference pseudocode:

    qualifying_shoppers = count of tracks inside ROI_queue with
                           dwell_seconds >= DWELL_TIME_SECONDS
    if qualifying_shoppers >= QUEUE_THRESHOLD:
        if continuous_congestion_time >= CONFIRMATION_WINDOW:
            dispatch_alert(...)

Congestion Index engine (M/M/c-style):
    lambda = observed arrival rate into ROI_queue (customers/minute)
    mu     = 1 / avg_service_time_seconds, expressed per minute
    CI     = lambda / (active_cashiers * mu)
    predicted_queue_depth(t + horizon) = people_count + (lambda - active_cashiers*mu) * horizon

This MVP uses the *observed* arrival rate as lambda_predicted -- the full
spec's feed-forward correction from upstream "final-stage shopping zone"
dwell (Module 3, Section 3) would need additional camera coverage that
isn't part of this camera layout, so it's a documented simplification
rather than an implemented feature.

Wait time uses the spec's simplified formula (Section 3.2):
    Wait Time ~= (People in Queue * Avg Service Time per Customer) / Active Cashiers
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import List

from app.geometry import Polygon
from app.vision.tracker import Track
from config import QueueConfig

RECOMMENDATION_OPTIMAL = "optimal"
RECOMMENDATION_WARNING = "warning"
RECOMMENDATION_CRITICAL = "critical_open_lanes"
RECOMMENDATION_IDLE = "idle_close_lane"


@dataclass
class QueueState:
    lane_id: str
    people_count: int
    qualifying_shoppers: int
    estimated_wait_seconds: float
    alert_triggered: bool
    continuous_congestion_seconds: float
    arrival_rate_per_minute: float
    service_rate_per_minute: float
    congestion_index: float
    predicted_queue_depth: float
    recommendation: str


class QueueMonitor:
    def __init__(self, lane_id: str, polygon: Polygon, config: QueueConfig,
                 active_cashiers: int = 1, avg_service_time_seconds: float = 90.0):
        self.lane_id = lane_id
        self.polygon = polygon
        self.config = config
        self.active_cashiers = active_cashiers
        self.avg_service_time_seconds = avg_service_time_seconds
        self._entered_at = {}           # track_id -> time entered ROI_queue
        self._congestion_since = None   # time qualifying_shoppers first breached threshold
        self._low_ci_since = None       # time congestion_index first dropped below the idle threshold
        self._arrival_events = deque()  # timestamps of new arrivals, for lambda estimation

    def evaluate(self, tracks: List[Track]) -> QueueState:
        now = time.time()
        in_polygon_ids = set()
        qualifying_shoppers = 0

        for track in tracks:
            if not self.polygon.contains(track.centroid):
                continue
            in_polygon_ids.add(track.track_id)
            if track.track_id not in self._entered_at:
                self._entered_at[track.track_id] = now
                self._arrival_events.append(now)
            dwell_seconds = now - self._entered_at[track.track_id]
            if dwell_seconds >= self.config.dwell_time_seconds:
                qualifying_shoppers += 1

        for track_id in list(self._entered_at.keys()):
            if track_id not in in_polygon_ids:
                del self._entered_at[track_id]

        while self._arrival_events and now - self._arrival_events[0] > self.config.arrival_window_seconds:
            self._arrival_events.popleft()

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

        arrival_rate_per_minute = len(self._arrival_events) / (self.config.arrival_window_seconds / 60.0)
        service_rate_per_minute = 60.0 / self.avg_service_time_seconds if self.avg_service_time_seconds > 0 else 0.0
        effective_service_capacity = self.active_cashiers * service_rate_per_minute
        congestion_index = (
            arrival_rate_per_minute / effective_service_capacity if effective_service_capacity > 0 else 0.0
        )

        horizon_minutes = self.config.forecast_horizon_seconds / 60.0
        predicted_queue_depth = max(
            0.0, people_count + (arrival_rate_per_minute - effective_service_capacity) * horizon_minutes
        )

        if congestion_index < self.config.ci_idle_threshold:
            if self._low_ci_since is None:
                self._low_ci_since = now
        else:
            self._low_ci_since = None
        idle_sustained = (
            self._low_ci_since is not None
            and now - self._low_ci_since >= self.config.ci_idle_sustained_seconds
        )

        if congestion_index >= self.config.ci_critical_threshold or predicted_queue_depth > self.config.threshold:
            recommendation = RECOMMENDATION_CRITICAL
        elif congestion_index >= self.config.ci_warning_threshold:
            recommendation = RECOMMENDATION_WARNING
        elif idle_sustained:
            recommendation = RECOMMENDATION_IDLE
        else:
            recommendation = RECOMMENDATION_OPTIMAL

        return QueueState(
            lane_id=self.lane_id,
            people_count=people_count,
            qualifying_shoppers=qualifying_shoppers,
            estimated_wait_seconds=estimated_wait_seconds,
            alert_triggered=alert_triggered,
            continuous_congestion_seconds=continuous_congestion_seconds,
            arrival_rate_per_minute=arrival_rate_per_minute,
            service_rate_per_minute=service_rate_per_minute,
            congestion_index=congestion_index,
            predicted_queue_depth=predicted_queue_depth,
            recommendation=recommendation,
        )
