"""
Geometric primitives used across the perception pipeline:
  - Tripwire crossing detection (Section 3.1: Entry/Exit Counting)
  - Polygon containment for queue / dwell ROIs (Section 3.2, Module 2)

No external geometry libraries are required; these are small, dependency-free
implementations so the "motion" tracker backend can run with only OpenCV/NumPy.
"""
from dataclasses import dataclass
from typing import List, Tuple

Point = Tuple[float, float]


@dataclass
class Tripwire:
    """A calibrated virtual tripwire across a doorway.

    p1 -> p2 defines the line. The sign of the cross product of the line
    vector against the vector to a point tells us which side of the line
    that point sits on; a sign flip between two consecutive frames means
    the tracked centroid crossed the wire.
    """
    p1: Point
    p2: Point
    name: str = "entrance"

    def side(self, point: Point) -> float:
        x1, y1 = self.p1
        x2, y2 = self.p2
        vx, vy = (x2 - x1, y2 - y1)
        wx, wy = (point[0] - x1, point[1] - y1)
        return vx * wy - vy * wx

    def crossing_direction(self, prev_point: Point, curr_point: Point) -> str:
        """Returns 'IN', 'OUT', or 'NONE' based on the sign flip of side()."""
        prev_side = self.side(prev_point)
        curr_side = self.side(curr_point)
        if prev_side == 0 or curr_side == 0:
            return "NONE"
        if (prev_side > 0) == (curr_side > 0):
            return "NONE"
        # Positive-to-negative crossing is defined as inbound.
        return "IN" if prev_side > 0 > curr_side else "OUT"


class Polygon:
    """Simple point-in-polygon test via the ray casting algorithm."""

    def __init__(self, points: List[Point], name: str = "roi"):
        self.points = points
        self.name = name

    def contains(self, point: Point) -> bool:
        x, y = point
        inside = False
        n = len(self.points)
        j = n - 1
        for i in range(n):
            xi, yi = self.points[i]
            xj, yj = self.points[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside


def euclidean_distance(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
