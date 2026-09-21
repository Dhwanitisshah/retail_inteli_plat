"""
Entry/Exit footfall counting via calibrated virtual tripwires
(Section 3.1: Entry/Exit Counting).
"""
import time
from collections import deque
from typing import List

from app.geometry import Tripwire
from app.vision.tracker import Track


class FootfallCounter:
    def __init__(self, tripwire: Tripwire, surge_window_seconds: float = 300.0):
        self.tripwire = tripwire
        self.surge_window_seconds = surge_window_seconds
        self._crossed_track_ids = set()
        self._inbound_events = deque()   # timestamps, for surge-rate calc
        self.total_in = 0
        self.total_out = 0

    def update(self, tracks: List[Track]):
        """Returns a list of ('IN' | 'OUT') events observed this frame."""
        events = []
        now = time.time()
        for track in tracks:
            if track.prev_centroid is None or track.track_id in self._crossed_track_ids:
                continue
            direction = self.tripwire.crossing_direction(track.prev_centroid, track.centroid)
            if direction == "NONE":
                continue
            self._crossed_track_ids.add(track.track_id)
            events.append(direction)
            if direction == "IN":
                self.total_in += 1
                self._inbound_events.append(now)
            else:
                self.total_out += 1

        self._prune_surge_window(now)
        # Bound memory: forget crossed IDs the tracker itself has long since dropped.
        if len(self._crossed_track_ids) > 5000:
            self._crossed_track_ids.clear()
        return events

    def _prune_surge_window(self, now: float):
        # Keep two full windows so surge_rate() can compare period-over-period.
        horizon = 2 * self.surge_window_seconds
        while self._inbound_events and now - self._inbound_events[0] > horizon:
            self._inbound_events.popleft()

    def surge_rate(self) -> float:
        """Fractional change in inbound corridor footfall between the previous
        window and the current one, e.g. 0.25 == a 25% surge (spec: '>20%
        surge rate' triggers a staffing alert alongside queue-length checks).
        Returns 0.0 until a full previous window of history exists."""
        now = time.time()
        window = self.surge_window_seconds
        current = sum(1 for t in self._inbound_events if t > now - window)
        previous = sum(1 for t in self._inbound_events if now - 2 * window < t <= now - window)
        if previous <= 0:
            return 0.0
        return (current - previous) / previous
