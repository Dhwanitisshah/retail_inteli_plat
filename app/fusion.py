"""
Fusion & priority engine -- the piece that turns "this slot looks empty"
into "here's why, and how urgent." Ported from the reference demo's cause
table and bounded priority formula (see the HTML's `fuse()` / `priorityFor()`):

Cause classification (checked in this order -- LOW velocity is checked
before backroom stock, so a gap with no sales never becomes an urgent
restock job):
    POS unavailable          -> VISUAL_ALERT_ONLY
    velocity class == LOW    -> LOW_PRIORITY_OOS   (likely misplacement, not a stockout)
    backroom has stock       -> REPLENISHMENT_REQUIRED
    backroom empty           -> PURCHASING_ALERT    (no floor task; reorder instead)

Priority score (bounded 0-100, bucketed P0-P3):
    raw   = (V / V_ref) * M * (G / G_ref) * A / (1 + D)
    score = clamp(raw / raw_ref, 0, 1) * 100

    V = sales velocity (units/h)             G = void_ratio * min(max(elapsed_min, floor), cap)
    M = per-SKU business-impact multiplier   A = 1.0 if backroom has stock else 0.15
    D = travel_penalty (0 = staff already in zone, >0 = out of zone)
"""
from dataclasses import dataclass
from typing import Optional

from app.pos import VELOCITY_LOW, VELOCITY_UNKNOWN

CAUSE_VISUAL_ONLY = "VISUAL_ALERT_ONLY"
CAUSE_REPLENISHMENT = "REPLENISHMENT_REQUIRED"
CAUSE_PURCHASING = "PURCHASING_ALERT"
CAUSE_LOW_PRIORITY = "LOW_PRIORITY_OOS"

BUCKET_THRESHOLDS = (("P0", 85), ("P1", 60), ("P2", 35), ("P3", 0))


def classify_cause(pos_available: bool, velocity_class: str, backroom_units: int) -> str:
    if not pos_available:
        return CAUSE_VISUAL_ONLY
    if velocity_class == VELOCITY_LOW:
        return CAUSE_LOW_PRIORITY
    if backroom_units > 0:
        return CAUSE_REPLENISHMENT
    return CAUSE_PURCHASING


def bucket_for_score(score: float) -> str:
    for label, minimum in BUCKET_THRESHOLDS:
        if score >= minimum:
            return label
    return "P3"


@dataclass
class PriorityResult:
    score: int
    bucket: str
    raw: float
    V: float
    G: float
    A: float
    D: float
    cause: str


def compute_priority(
    velocity_per_hour: Optional[float],
    baseline_per_hour: float,
    business_impact_multiplier: float,
    void_ratio: float,
    gap_elapsed_seconds: float,
    backroom_units: int,
    cause: str,
    travel_penalty: float = 0.0,
    v_ref: float = 6.0,
    g_ref: float = 5.0,
    raw_ref: float = 4.5,
    duration_floor_minutes: float = 5.0,
    duration_cap_minutes: float = 60.0,
) -> PriorityResult:
    V = velocity_per_hour if velocity_per_hour is not None else baseline_per_hour
    elapsed_minutes = max(0.0, gap_elapsed_seconds / 60.0)
    duration = min(duration_cap_minutes, max(duration_floor_minutes, elapsed_minutes))
    G = void_ratio * duration
    A = 1.0 if backroom_units > 0 else 0.15
    D = travel_penalty

    raw = (V / v_ref) * business_impact_multiplier * (G / g_ref) * A / (1 + D)
    score = max(0.0, min(1.0, raw / raw_ref)) * 100.0
    return PriorityResult(
        score=round(score), bucket=bucket_for_score(score), raw=raw,
        V=V, G=G, A=A, D=D, cause=cause,
    )
