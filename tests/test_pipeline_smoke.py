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
from app.fusion import (
    CAUSE_LOW_PRIORITY, CAUSE_PURCHASING, CAUSE_REPLENISHMENT, CAUSE_VISUAL_ONLY,
    classify_cause, compute_priority,
)
from app.geometry import Polygon, Tripwire
from app.inventory import MockInventoryAdapter
from app.pos import VELOCITY_HIGH, VELOCITY_LOW, VELOCITY_UNKNOWN, MockPOSAdapter
from app.roster import Roster, Staff
from app.tasks import RESOLUTION_RESOLVED, RESOLUTION_VERIFY_UNAVAILABLE, TaskManager
from app.vision.footfall import FootfallCounter
from app.vision.heatmap import HeatmapAccumulator
from app.vision.homography import PlanarHomography
from app.vision.ocr import LabelReader
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


def test_congestion_index_critical_recommendation():
    polygon = Polygon(points=[(0, 0), (500, 0), (500, 500), (0, 500)], name="counter_02")
    cfg = QueueConfig(
        threshold=100, dwell_time_seconds=0.0, confirmation_window=999999.0,
        arrival_window_seconds=60.0, forecast_horizon_seconds=300.0,
        ci_warning_threshold=0.70, ci_critical_threshold=1.00,
    )
    monitor = QueueMonitor(lane_id="counter_02", polygon=polygon, config=cfg,
                            active_cashiers=1, avg_service_time_seconds=30.0)  # mu = 2/min

    # 6 distinct shoppers arriving in one evaluation -> lambda = 6 / (60s/60) = 6/min
    shoppers = [make_track(100 + i, prev=(10, 10), curr=(10, 10)) for i in range(6)]
    state = monitor.evaluate(shoppers)

    assert abs(state.arrival_rate_per_minute - 6.0) < 1e-6, state.arrival_rate_per_minute
    assert abs(state.service_rate_per_minute - 2.0) < 1e-6, state.service_rate_per_minute
    assert abs(state.congestion_index - 3.0) < 1e-6, state.congestion_index
    assert state.recommendation == "critical_open_lanes"
    print("PASS: congestion index formula (CI=lambda/(c*mu)) + critical recommendation tier")


def test_congestion_index_idle_recommendation():
    polygon = Polygon(points=[(0, 0), (100, 0), (100, 100), (0, 100)], name="counter_03")
    cfg = QueueConfig(
        threshold=100, dwell_time_seconds=0.0, confirmation_window=999999.0,
        arrival_window_seconds=60.0, ci_idle_threshold=0.35, ci_idle_sustained_seconds=0.0,
    )
    monitor = QueueMonitor(lane_id="counter_03", polygon=polygon, config=cfg,
                            active_cashiers=1, avg_service_time_seconds=60.0)

    state = monitor.evaluate([])  # nobody in the lane
    assert state.congestion_index == 0.0
    assert state.recommendation == "idle_close_lane"
    print("PASS: congestion index idle/close-lane recommendation tier")


def test_homography_projection_scales_distance():
    # 100x100 px square mapped onto a 50x50 cm square -> uniform 0.5 cm/px scale
    pixel_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
    metric_points = [(0, 0), (50, 0), (50, 50), (0, 50)]
    homography = PlanarHomography(pixel_points, metric_points)

    distance_cm = homography.project_distance_cm((0, 50), (100, 50))
    assert abs(distance_cm - 50.0) < 1e-3, distance_cm
    print("PASS: planar homography pixel->cm projection (x' = Hx)")


def test_shelf_void_gap_cm_with_homography():
    slot = ShelfSlot(aisle_id="AISLE_01", shelf_tier=1, slot_id="A1_T1_S1",
                      sku_id="SKU_TEST", rect=(0, 0, 200, 100), expected_facings=3)
    cfg = ShelfConfig(fill_ratio_warn=0.30, void_ratio_alert=0.5, consecutive_frames_required=1)
    pixel_points = [(0, 0), (200, 0), (200, 100), (0, 100)]
    metric_points = [(0, 0), (100, 0), (100, 50), (0, 50)]  # 0.5 cm/px
    homography = PlanarHomography(pixel_points, metric_points)
    detector = ShelfVoidDetector([slot], cfg, homography=homography)

    stocked_frame = build_slot_frame(200, 100, slot.rect, textured=True)
    detector.calibrate(stocked_frame)
    empty_frame = build_slot_frame(200, 100, slot.rect, textured=False)  # whole 200px-wide slot is void

    state = detector.evaluate(empty_frame)[0]
    assert state.gap_cm is not None
    assert abs(state.gap_cm - 100.0) < 1.0, state.gap_cm  # 200px * 0.5cm/px == 100cm
    print("PASS: shelf void gap measured in metric cm via homography")


def test_heatmap_accumulator_updates_and_renders():
    heatmap = HeatmapAccumulator(width=200, height=100, cell_size=10)
    track = make_track(1, prev=(50, 50), curr=(50, 50))
    for _ in range(5):
        heatmap.update([track])

    png_bytes = heatmap.render_png_bytes("traffic")
    assert len(png_bytes) > 0
    gy, gx = 50 // 10, 50 // 10
    assert heatmap.traffic[gy, gx] > 0
    print("PASS: heatmap accumulator updates + PNG render")


def test_ocr_graceful_fallback():
    reader = LabelReader()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = reader.read(frame, (0, 0, 50, 50))
    # No Tesseract binary is guaranteed to exist in the test environment, so this
    # must degrade to None without raising regardless of `reader.available`.
    assert result is None or hasattr(result, "text")
    print(f"PASS: OCR module never raises (available={reader.available})")


def test_pos_adapter_velocity_classification():
    pos = MockPOSAdapter(window_seconds=3600.0, high_multiplier=1.5, low_multiplier=0.5)
    # 0 sales against a real baseline is legitimately LOW, not UNKNOWN -- UNKNOWN is
    # reserved for "no trustworthy baseline" or "POS unavailable" (checked below).
    assert pos.velocity_class("sku_a", baseline_per_hour=10.0) == VELOCITY_LOW

    pos.record_sale("sku_a", 20)  # 20 units in the last hour -> 20/h vs baseline 10/h -> ratio 2.0 -> HIGH
    assert pos.velocity_class("sku_a", baseline_per_hour=10.0) == VELOCITY_HIGH

    pos.set_available(False)
    assert pos.velocity_per_hour("sku_a") is None
    assert pos.velocity_class("sku_a", baseline_per_hour=10.0) == VELOCITY_UNKNOWN
    pos.set_available(True)

    assert pos.velocity_class("sku_b", baseline_per_hour=0.5) == VELOCITY_UNKNOWN, "baseline below min_baseline_units must never resolve to LOW"
    print("PASS: mock POS adapter velocity classification (HIGH / UNKNOWN on outage / UNKNOWN on thin baseline)")


def test_inventory_adapter_transfer_bounds():
    inv = MockInventoryAdapter(backroom={"sku_a": 5}, system={"sku_a": 20})
    assert inv.get_backroom_units("sku_a") == 5
    assert inv.get_backroom_units("sku_missing") == 0

    moved = inv.transfer_to_shelf("sku_a", requested_units=3)
    assert moved == 3 and inv.get_backroom_units("sku_a") == 2

    moved = inv.transfer_to_shelf("sku_a", requested_units=99)
    assert moved == 2 and inv.get_backroom_units("sku_a") == 0, "transfer must never exceed what's actually in backroom"
    print("PASS: mock inventory adapter backroom transfer bounds")


def test_fusion_cause_classification():
    assert classify_cause(pos_available=False, velocity_class=VELOCITY_HIGH, backroom_units=5) == CAUSE_VISUAL_ONLY
    assert classify_cause(pos_available=True, velocity_class=VELOCITY_LOW, backroom_units=5) == CAUSE_LOW_PRIORITY, \
        "LOW velocity must win even with backroom stock -- likely misplacement, not a stockout"
    assert classify_cause(pos_available=True, velocity_class=VELOCITY_HIGH, backroom_units=5) == CAUSE_REPLENISHMENT
    assert classify_cause(pos_available=True, velocity_class=VELOCITY_HIGH, backroom_units=0) == CAUSE_PURCHASING
    print("PASS: fusion cause classification truth table")


def test_fusion_priority_bounds_and_bucket():
    low = compute_priority(velocity_per_hour=0.5, baseline_per_hour=2.0, business_impact_multiplier=1.0,
                            void_ratio=0.7, gap_elapsed_seconds=60, backroom_units=0, cause=CAUSE_PURCHASING)
    high = compute_priority(velocity_per_hour=50.0, baseline_per_hour=2.0, business_impact_multiplier=1.3,
                             void_ratio=1.0, gap_elapsed_seconds=3600, backroom_units=10, cause=CAUSE_REPLENISHMENT)
    for result in (low, high):
        assert 0 <= result.score <= 100, result.score
    assert high.score > low.score, "a fast-selling, long-empty, in-stock-backroom slot must outrank a slow one"
    assert high.bucket in ("P0", "P1")
    assert low.bucket in ("P2", "P3")
    print("PASS: fusion priority score bounded [0,100] and correctly bucketed")


def test_roster_assignment_scoring():
    roster = Roster([
        Staff(staff_id="S1", name="Alice", zones=["AISLE_04"]),
        Staff(staff_id="S2", name="Bob", zones=["CHECKOUT"]),
    ])
    chosen = roster.best_candidate("AISLE_04")
    assert chosen.staff_id == "S1", "the only in-zone, idle candidate should win"

    roster.enqueue("S1", "T-0001")  # Alice is now busy
    chosen = roster.best_candidate("AISLE_04")
    assert chosen.staff_id == "S2", \
        "an idle out-of-zone staffer can outscore a busy in-zone one (0.5 idle-out-of-zone > 0.175 busy-in-zone)"

    roster.release("S1", "T-0001")
    assert roster.staff["S1"].active_task_id is None
    chosen = roster.best_candidate("AISLE_04")
    assert chosen.staff_id == "S1", "once freed, the in-zone idle staffer wins again"
    print("PASS: roster assignment scoring (in-zone vs. workload trade-off)")


def test_task_lifecycle_resolved_by_camera():
    roster = Roster([Staff(staff_id="S1", name="Alice", zones=["AISLE_04"])])
    tm = TaskManager(roster, sla_minutes={"restock": 60}, verify_after_seconds=0.0,
                      recheck_interval_seconds=0.0, max_verify_attempts=3)
    priority = compute_priority(velocity_per_hour=5.0, baseline_per_hour=2.0, business_impact_multiplier=1.0,
                                 void_ratio=0.9, gap_elapsed_seconds=300, backroom_units=8, cause=CAUSE_REPLENISHMENT)
    task = tm.create_task(kind="restock", zone="AISLE_04", title="Restock sugar", reason="void 0.90",
                           cause=CAUSE_REPLENISHMENT, priority=priority, slot_id="A4_T2_S1", sku_id="SKU_TEST")
    assert task.staff_id == "S1" and task.status == "ASSIGNED"

    tm.mark_completed(task.task_id)
    assert task.status == "VERIFYING"

    tm.tick(is_resolved=lambda t: True)  # camera now sees the shelf restocked
    assert task.resolution_state == RESOLUTION_RESOLVED and task.status == "CLOSED"
    assert roster.staff["S1"].active_task_id is None, "staff must be freed once the task closes"
    print("PASS: task lifecycle resolves when the camera confirms the fix")


def test_task_lifecycle_escalates_to_verify_unavailable():
    roster = Roster([Staff(staff_id="S1", name="Alice", zones=["AISLE_04"])])
    tm = TaskManager(roster, sla_minutes={"restock": 60}, verify_after_seconds=0.0,
                      recheck_interval_seconds=0.0, max_verify_attempts=2)
    priority = compute_priority(velocity_per_hour=5.0, baseline_per_hour=2.0, business_impact_multiplier=1.0,
                                 void_ratio=0.9, gap_elapsed_seconds=300, backroom_units=8, cause=CAUSE_REPLENISHMENT)
    task = tm.create_task(kind="restock", zone="AISLE_04", title="Restock sugar", reason="void 0.90",
                           cause=CAUSE_REPLENISHMENT, priority=priority, slot_id="A4_T2_S1", sku_id="SKU_TEST")
    tm.mark_completed(task.task_id)

    always_unresolved = lambda t: False
    for _ in range(2):
        tm.tick(is_resolved=always_unresolved)
    assert task.resolution_state == RESOLUTION_VERIFY_UNAVAILABLE and task.status == "CLOSED"
    assert task.verify_attempts == 2
    print("PASS: task escalates to VERIFY_UNAVAILABLE after exhausting retries, never silently 'resolved'")


def test_task_sla_breach_flag():
    roster = Roster([Staff(staff_id="S1", name="Alice", zones=["AISLE_04"])])
    tm = TaskManager(roster, sla_minutes={"restock": 0}, verify_after_seconds=999999.0)
    priority = compute_priority(velocity_per_hour=5.0, baseline_per_hour=2.0, business_impact_multiplier=1.0,
                                 void_ratio=0.9, gap_elapsed_seconds=300, backroom_units=8, cause=CAUSE_REPLENISHMENT)
    task = tm.create_task(kind="restock", zone="AISLE_04", title="Restock sugar", reason="void 0.90",
                           cause=CAUSE_REPLENISHMENT, priority=priority, slot_id="A4_T2_S1", sku_id="SKU_TEST")
    assert task.sla_breached is False
    tm.tick(is_resolved=lambda t: False)
    assert task.sla_breached is True, "0-minute SLA must already be breached on the first tick"
    assert task.status != "CLOSED", "a breach alone doesn't close the task -- it can still resolve, just late"
    print("PASS: SLA-deadline breach flag set independently of verification state")


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
    test_congestion_index_critical_recommendation()
    test_congestion_index_idle_recommendation()
    test_homography_projection_scales_distance()
    test_shelf_void_detection()
    test_shelf_occlusion_gating()
    test_shelf_void_gap_cm_with_homography()
    test_heatmap_accumulator_updates_and_renders()
    test_ocr_graceful_fallback()
    test_pos_adapter_velocity_classification()
    test_inventory_adapter_transfer_bounds()
    test_fusion_cause_classification()
    test_fusion_priority_bounds_and_bucket()
    test_roster_assignment_scoring()
    test_task_lifecycle_resolved_by_camera()
    test_task_lifecycle_escalates_to_verify_unavailable()
    test_task_sla_breach_flag()
    test_database_persistence()
    test_alert_dispatcher_fanout()
    print("\nAll smoke tests passed.")
