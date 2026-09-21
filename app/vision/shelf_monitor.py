"""
Class-Agnostic Shelf Void Detection & Planogram Mapping (Section 3.3).

The full spec calls for a trained YOLOv8-Seg instance-segmentation model that
outputs `product_cluster` / `empty_shelf_void` masks per slot. Training that
model requires a labeled shelf-imagery dataset that isn't available in this
environment, so this module implements a classical-CV stand-in that follows
the same geometric contract (per-slot void_ratio via calibrated reference
comparison, and vertical fill_ratio) so it can be swapped for the real
segmentation model later without touching any downstream code (alerting,
persistence, planogram mapping).

Heuristic: a bare/empty shelf slot (metal deck, pegboard, dividers) has much
lower edge density than a slot packed with product packaging. Each slot is
calibrated once against a known fully-stocked reference frame; void_ratio is
then how far current edge density has dropped relative to that baseline.

Human Occlusion Filter (Module 4, Section 1): if a person track's bounding
box significantly overlaps a slot, that slot's evaluation is skipped for the
frame and its last known state is reused, so a shopper reaching for an item
is never misread as an empty shelf.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from config import ShelfConfig


@dataclass
class ShelfSlot:
    aisle_id: str
    shelf_tier: int
    slot_id: str
    sku_id: str
    rect: Tuple[int, int, int, int]  # x, y, w, h in pixel coordinates
    expected_facings: int = 1


@dataclass
class ShelfSlotState:
    slot: ShelfSlot
    void_ratio: float
    fill_ratio: float
    low_stock_warning: bool
    out_of_stock_alert: bool
    consecutive_alert_frames: int
    occluded: bool = False


def _bbox_overlap_ratio(rect: Tuple[int, int, int, int], bbox: Tuple[int, int, int, int]) -> float:
    rx, ry, rw, rh = rect
    bx, by, bw, bh = bbox
    ix1, iy1 = max(rx, bx), max(ry, by)
    ix2, iy2 = min(rx + rw, bx + bw), min(ry + rh, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    slot_area = rw * rh
    return intersection / slot_area if slot_area > 0 else 0.0


class ShelfVoidDetector:
    def __init__(self, slots: List[ShelfSlot], config: ShelfConfig, occlusion_threshold: float = 0.25):
        self.slots = slots
        self.config = config
        self.occlusion_threshold = occlusion_threshold
        self._reference_edge_density: Dict[str, float] = {}
        self._consecutive_over_threshold: Dict[str, int] = {s.slot_id: 0 for s in slots}
        self._last_state: Dict[str, ShelfSlotState] = {}
        self._calibrated = False

    def calibrate(self, frame):
        """Run once against a frame of a known fully-stocked shelf to set the
        product-cluster edge-density baseline for every slot."""
        for slot in self.slots:
            density = self._edge_density(frame, slot.rect)
            self._reference_edge_density[slot.slot_id] = max(density, 0.01)
        self._calibrated = True

    @staticmethod
    def _edge_density(frame, rect) -> float:
        x, y, w, h = rect
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(np.count_nonzero(edges)) / edges.size

    @staticmethod
    def _vertical_fill_ratio(frame, rect) -> float:
        x, y, w, h = rect
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        row_activity = edges.sum(axis=1)
        active_rows = np.where(row_activity > (edges.shape[1] * 255 * 0.03))[0]
        if len(active_rows) == 0:
            return 0.0
        stack_height = active_rows[-1] - active_rows[0] + 1
        return min(1.0, float(stack_height) / h)

    def evaluate(self, frame, person_bboxes: List[Tuple[int, int, int, int]] = None) -> List[ShelfSlotState]:
        if not self._calibrated:
            self.calibrate(frame)
        person_bboxes = person_bboxes or []

        results = []
        for slot in self.slots:
            occluded = any(
                _bbox_overlap_ratio(slot.rect, bbox) >= self.occlusion_threshold
                for bbox in person_bboxes
            )
            if occluded and slot.slot_id in self._last_state:
                cached = self._last_state[slot.slot_id]
                results.append(ShelfSlotState(
                    slot=slot, void_ratio=cached.void_ratio, fill_ratio=cached.fill_ratio,
                    low_stock_warning=cached.low_stock_warning,
                    out_of_stock_alert=cached.out_of_stock_alert,
                    consecutive_alert_frames=cached.consecutive_alert_frames,
                    occluded=True,
                ))
                continue

            density = self._edge_density(frame, slot.rect)
            reference = self._reference_edge_density.get(slot.slot_id, 0.05)
            void_ratio = max(0.0, min(1.0, 1.0 - (density / reference)))
            fill_ratio = self._vertical_fill_ratio(frame, slot.rect)

            if void_ratio > self.config.void_ratio_alert:
                self._consecutive_over_threshold[slot.slot_id] += 1
            else:
                self._consecutive_over_threshold[slot.slot_id] = 0

            consecutive = self._consecutive_over_threshold[slot.slot_id]
            state = ShelfSlotState(
                slot=slot,
                void_ratio=void_ratio,
                fill_ratio=fill_ratio,
                low_stock_warning=fill_ratio < self.config.fill_ratio_warn,
                out_of_stock_alert=consecutive >= self.config.consecutive_frames_required,
                consecutive_alert_frames=consecutive,
                occluded=False,
            )
            self._last_state[slot.slot_id] = state
            results.append(state)
        return results
