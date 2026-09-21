"""
End-to-end smoke test for the perception + persistence + alerting pipeline.
Runs without pytest so it only needs opencv/numpy (already in requirements.txt):

    python tests/test_pipeline_smoke.py
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.alerts.dispatcher import AlertDispatcher, AlertEvent
from app.database import Database
from app.geometry import Polygon, Tripwire
from app.vision.footfall import FootfallCounter
from app.vision.queue_monitor import QueueMonitor
from app.vision.shelf_monitor import ShelfSlot, ShelfVoidDetector
from app.vision.tracker import Track
from config import AlertConfig, QueueConfig, ShelfConfig


def make_track(track_id, prev, curr, first_seen=None, last_seen=None):
    now = time.time()
    return Track(
        track_id=track_id, centroid=curr, bbox=(int(curr[0]) - 10, int(curr[1]) - 10, 20, 20),
        first_seen=first_seen or now, last_seen=last_seen or now, prev_centroid=prev,
    )


def test_footfall_tripwire_in_and_out():
    tripwire = Tripwire(p1=(0, 100), p2=(200, 100), name="test_entrance")
    counter = FootfallCounter(tripwire, surge_window_seconds=300)

    # side() > 0 below the line (y > 100), side() < 0 above it (y < 100);
    # positive-to-negative is defined as IN (see geometry.Tripwire docstring).
    inbound_track = make_track(1, prev=(50, 150), curr=(50, 50))
    events = counter.update([inbound_track])
    assert events == ["IN"], f"expected IN crossing, got {events}"
    assert counter.total_in == 1 and counter.total_out == 0

    outbound_track = make_track(2, prev=(50, 50), curr=(50, 150))
    events = counter.update([outbound_track])
    assert events == ["OUT"], f"expected OUT crossing, got {events}"
    assert counter.total_out == 1

    # Re-evaluating a track id that already crossed shouldn't double count.
    events = counter.update([make_track(1, prev=(50, 50), curr=(50, 50))])
    assert events == [], "already-crossed track id should not be recounted"
    print("PASS: footfall tripwire IN/OUT crossing + de-dup")


def test_queue_alert_debounce():
    polygon = Polygon(points=[(0, 0), (100, 0), (100, 100), (0, 100)], name="counter_01")
    # Zeroed thresholds so a single evaluate() call deterministically exercises
    # the alert path without sleeping in a unit test.
    cfg = QueueConfig(threshold=1, dwell_time_seconds=0.0, confirmation_window=0.0, surge_rate=0.20)
    monitor = QueueMonitor(lane_id="counter_01", polygon=polygon, config=cfg,
                            active_cashiers=1, avg_service_time_seconds=90.0)

    shopper = make_track(10, prev=(50, 50), curr=(50, 50))
    state = monitor.evaluate([shopper])
    assert state.people_count == 1
    assert state.qualifying_shoppers == 1
    assert state.alert_triggered is True, "expected immediate alert with zeroed debounce thresholds"

    # Shopper leaves the ROI -> no longer qualifies, alert clears.
    state = monitor.evaluate([make_track(10, prev=(50, 50), curr=(500, 500))])
    assert state.people_count == 0
    assert state.alert_triggered is False
    print("PASS: queue dwell-time + confirmation-window alerting")


def build_slot_frame(width, height, slot_rect, textured: bool):
    frame = np.full((height, width, 3), 60, dtype=np.uint8)  # flat bare-shelf background
    x, y, w, h = slot_rect
    if textured:
        rng = np.random.default_rng(42)
        frame[y:y + h, x:x + w] = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return frame


def test_shelf_void_detection():
    slot = ShelfSlot(aisle_id="AISLE_01", shelf_tier=1, slot_id="A1_T1_S1",
                      sku_id="SKU_TEST", rect=(50, 50, 100, 100), expected_facings=3)
    cfg = ShelfConfig(fill_ratio_warn=0.30, void_ratio_alert=0.5, consecutive_frames_required=2)
    detector = ShelfVoidDetector([slot], cfg)

    stocked_frame = build_slot_frame(300, 200, slot.rect, textured=True)
    detector.calibrate(stocked_frame)

    empty_frame = build_slot_frame(300, 200, slot.rect, textured=False)

    state1 = detector.evaluate(empty_frame)[0]
    assert state1.void_ratio > cfg.void_ratio_alert, f"expected high void_ratio, got {state1.void_ratio}"
    assert state1.out_of_stock_alert is False, "should not alert before consecutive_frames_required is met"

    state2 = detector.evaluate(empty_frame)[0]
    assert state2.out_of_stock_alert is True, "should alert after 2 consecutive over-threshold frames"

    # Restocked -> counter resets and alert clears.
    restocked_frame = build_slot_frame(300, 200, slot.rect, textured=True)
    state3 = detector.evaluate(restocked_frame)[0]
    assert state3.void_ratio < cfg.void_ratio_alert
    assert state3.out_of_stock_alert is False
    print("PASS: shelf void detection with consecutive-frame debounce")


def test_shelf_occlusion_gating():
    slot = ShelfSlot(aisle_id="AISLE_01", shelf_tier=1, slot_id="A1_T1_S1",
                      sku_id="SKU_TEST", rect=(50, 50, 100, 100), expected_facings=3)
    cfg = ShelfConfig(void_ratio_alert=0.5, consecutive_frames_required=1)
    detector = ShelfVoidDetector([slot], cfg, occlusion_threshold=0.25)

    stocked_frame = build_slot_frame(300, 200, slot.rect, textured=True)
    detector.calibrate(stocked_frame)
    first = detector.evaluate(stocked_frame)[0]
    assert first.occluded is False

    empty_frame = build_slot_frame(300, 200, slot.rect, textured=False)
    person_bbox = (50, 50, 80, 80)  # overlaps 64% of the slot area
    occluded_state = detector.evaluate(empty_frame, person_bboxes=[person_bbox])[0]
    assert occluded_state.occluded is True
    assert occluded_state.void_ratio == first.void_ratio, "occluded slot should reuse last known state"
    print("PASS: shelf human-occlusion gating")


def test_database_persistence():
    tmp_dir = tempfile.mkdtemp(prefix="retail_edge_test_")
    db_path = os.path.join(tmp_dir, "test.db")
    db = Database(db_path)

    db.insert_footfall_event("IN")
    db.insert_footfall_event("OUT")
    recent = db.recent_footfall(limit=10)
    assert len(recent) == 2

    db.insert_queue_telemetry("counter_01", people_count=3, estimated_wait_seconds=120.0)
    telemetry = db.recent_queue_telemetry("counter_01")
    assert len(telemetry) == 1 and telemetry[0]["people_count"] == 3

    db.insert_shelf_event("AISLE_01", 1, "A1_T1_S1", "SKU_TEST", void_ratio=0.9, alert_triggered=True)
    alerts = db.active_shelf_alerts()
    assert len(alerts) == 1 and alerts[0]["slot_id"] == "A1_T1_S1"
    print("PASS: SQLite persistence (footfall / queue / shelf tables)")


def test_alert_dispatcher_fanout():
    received = []
    dispatcher = AlertDispatcher(AlertConfig(telegram_bot_token="", telegram_chat_id="", webhook_url=""))
    dispatcher.register_listener(lambda evt: received.append(evt))
    dispatcher.dispatch(AlertEvent(event_type="TEST_EVENT", priority="P1", message="hello"))
    assert len(received) == 1 and received[0].event_type == "TEST_EVENT"
    print("PASS: alert dispatcher in-process listener fanout")


if __name__ == "__main__":
    test_footfall_tripwire_in_and_out()
    test_queue_alert_debounce()
    test_shelf_void_detection()
    test_shelf_occlusion_gating()
    test_database_persistence()
    test_alert_dispatcher_fanout()
    print("\nAll smoke tests passed.")
