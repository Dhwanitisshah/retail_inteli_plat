"""
POS adapter (mock). The reference demo pulls sales velocity and a checkout
service rate (mu) from a real POS's transaction log. This repo has no POS
integration, so `MockPOSAdapter` records sale events in memory and computes
the same two things a real adapter would expose: a rolling units/hour
velocity classified against a per-SKU baseline (HIGH / NORMAL / LOW, same
1.5x / 0.5x multipliers as the reference), and unavailability (so callers
can fall back to VISUAL_ALERT_ONLY exactly like a real POS outage would
force).
"""
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

VELOCITY_HIGH = "HIGH"
VELOCITY_NORMAL = "NORMAL"
VELOCITY_LOW = "LOW"
VELOCITY_UNKNOWN = "UNKNOWN"


@dataclass
class SaleEvent:
    sku_id: str
    units: int
    timestamp: float


class POSAdapter(ABC):
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def record_sale(self, sku_id: str, units: int) -> None: ...

    @abstractmethod
    def velocity_per_hour(self, sku_id: str) -> Optional[float]:
        """Units sold in the trailing window, normalized to units/hour.
        None if POS is unavailable."""

    @abstractmethod
    def velocity_class(self, sku_id: str, baseline_per_hour: float) -> str:
        """HIGH / NORMAL / LOW / UNKNOWN -- UNKNOWN whenever POS is down or
        there isn't enough history to trust a comparison, never silently
        treated as LOW (a missing signal should never suppress an alert)."""


class MockPOSAdapter(POSAdapter):
    def __init__(self, window_seconds: float = 3600.0, high_multiplier: float = 1.5,
                 low_multiplier: float = 0.5, min_baseline_units: float = 1.0):
        self.window_seconds = window_seconds
        self.high_multiplier = high_multiplier
        self.low_multiplier = low_multiplier
        self.min_baseline_units = min_baseline_units
        self._available = True
        self._events: Dict[str, deque] = {}

    def set_available(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def record_sale(self, sku_id: str, units: int) -> None:
        if units <= 0:
            return
        bucket = self._events.setdefault(sku_id, deque())
        bucket.append(SaleEvent(sku_id=sku_id, units=units, timestamp=time.time()))
        self._prune(bucket)

    def _prune(self, bucket: deque) -> None:
        cutoff = time.time() - self.window_seconds
        while bucket and bucket[0].timestamp < cutoff:
            bucket.popleft()

    def velocity_per_hour(self, sku_id: str) -> Optional[float]:
        if not self._available:
            return None
        bucket = self._events.get(sku_id)
        if not bucket:
            return 0.0
        self._prune(bucket)
        total_units = sum(e.units for e in bucket)
        return total_units * (3600.0 / self.window_seconds)

    def velocity_class(self, sku_id: str, baseline_per_hour: float) -> str:
        v = self.velocity_per_hour(sku_id)
        if v is None:
            return VELOCITY_UNKNOWN
        if baseline_per_hour < self.min_baseline_units:
            return VELOCITY_UNKNOWN
        ratio = v / baseline_per_hour
        if ratio >= self.high_multiplier:
            return VELOCITY_HIGH
        if ratio < self.low_multiplier:
            return VELOCITY_LOW
        return VELOCITY_NORMAL
