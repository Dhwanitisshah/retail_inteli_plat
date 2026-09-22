"""
Entry point: wires the edge vision pipeline(s) to persistence, alerting, and
the local FastAPI dashboard/API server. Mirrors Section 2 of the spec end to
end, minus hardware-specific pieces (TensorRT, GStreamer/NVDEC) that only
make sense on real edge silicon -- everything here runs on plain OpenCV so
it's demoable on a laptop, with a pluggable YOLOv8+ByteTrack backend for
production hardware (see app/vision/tracker.py).

Usage:
    python main.py                                # run every camera in the layout + dashboard
    python main.py --only entrance_cam             # one physical webcam backs one camera key at a time
    python main.py --entrance-source demo.mp4 --only entrance_cam --loop
    python main.py --detector yolo                 # swap in the production YOLOv8+ByteTrack backend
    python main.py --show                          # open local debug overlay window(s)
    python main.py --show --ocr                    # also overlay shelf-tag OCR text (needs pytesseract + Tesseract)
    python main.py --no-server                     # run the vision pipeline only, no dashboard/API
"""
import argparse
import json
import logging
import threading
import time
from dataclasses import dataclass

import cv2
import uvicorn

from app import fusion
from app.alerts.dispatcher import AlertDispatcher, AlertEvent, PRIORITY_P1, PRIORITY_P2, PRIORITY_P4
from app.database import Database
from app.fusion import PriorityResult
from app.geometry import Polygon, Tripwire
from app.inventory import MockInventoryAdapter
from app.pos import MockPOSAdapter
from app.roster import Roster, Staff
from app.server.api import create_app
from app.state import LIVE_STATE
from app.tasks import TaskManager
from app.vision.footfall import FootfallCounter
from app.vision.heatmap import HeatmapAccumulator
from app.vision.homography import ArucoHomographyCalibrator, PlanarHomography
from app.vision.ocr import LabelReader
from app.vision.overlay import draw_polygons, draw_shelf_slots, draw_tracks, draw_tripwires
from app.vision.queue_monitor import (
    QueueMonitor, RECOMMENDATION_CRITICAL, RECOMMENDATION_IDLE, RECOMMENDATION_WARNING,
)
from app.vision.shelf_monitor import ShelfSlot, ShelfVoidDetector
from app.vision.stream import FrameSource
from app.vision.tracker import build_tracker
from config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")


@dataclass
class Services:
    """Everything downstream of raw detection: persistence, alerting, and the
    fusion/task subsystem (app/fusion.py, app/tasks.py) that turns a shelf
    void or a congestion forecast into an assigned, tracked, camera-verified
    task -- the piece the original MVP was missing (it could alert, but
    never knew whether anyone fixed anything)."""
    db: Database
    dispatcher: AlertDispatcher
    inventory: MockInventoryAdapter
    pos: MockPOSAdapter
    tasks: TaskManager


def load_layout(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_roster(layout) -> Roster:
    staff = [Staff(staff_id=s["staff_id"], name=s["name"], zones=s["zones"]) for s in layout.get("roster", [])]
    return Roster(staff)


def build_inventory_adapter(cameras) -> MockInventoryAdapter:
    backroom = {}
    for cam_config in cameras.values():
        for s in cam_config.get("shelf_slots", []):
            backroom[s["sku_id"]] = s.get("backroom_units", 0)
    return MockInventoryAdapter(backroom=backroom)


def make_is_resolved():
    """Verification predicate for TaskManager.tick(): re-checks the *live*
    camera/queue signal that originally raised the alert, never a staff
    self-report -- a task only resolves when the thing that noticed it says
    it's gone."""
    def is_resolved(task):
        snapshot = LIVE_STATE.snapshot()
        if task.slot_id:
            slot = snapshot["shelf_slots"].get(task.slot_id)
            return slot is not None and not slot.get("out_of_stock_alert", False)
        if task.lane_id:
            queue = snapshot["queues"].get(task.lane_id)
            return queue is not None and queue.get("recommendation") != RECOMMENDATION_CRITICAL
        return False
    return is_resolved


def run_task_ticker(task_manager: TaskManager, stop_event: threading.Event, interval: float = 2.0):
    """Single dedicated thread for TaskManager.tick() -- verification and
    SLA-deadline checks must not run concurrently from multiple camera
    pipeline threads, since they mutate shared task state."""
    is_resolved = make_is_resolved()
    logger.info("[tasks] verification ticker started (every %.0fs)", interval)
    while not stop_event.is_set():
        try:
            task_manager.tick(is_resolved)
        except Exception:
            logger.exception("[tasks] tick() failed")
        stop_event.wait(interval)


def build_footfall_counters(cam_config, footfall_cfg):
    counters = []
    for tw in cam_config.get("tripwires", []):
        tripwire = Tripwire(p1=tuple(tw["p1"]), p2=tuple(tw["p2"]), name=tw["name"])
        counters.append(FootfallCounter(tripwire, surge_window_seconds=footfall_cfg.surge_window_seconds))
    return counters


def build_queue_monitors(cam_config, queue_cfg):
    monitors = []
    for q in cam_config.get("queues", []):
        polygon = Polygon(points=[tuple(p) for p in q["polygon"]], name=q["lane_id"])
        monitors.append(QueueMonitor(
            lane_id=q["lane_id"], polygon=polygon, config=queue_cfg,
            active_cashiers=q.get("active_cashiers", 1),
            avg_service_time_seconds=q.get("avg_service_time_seconds", 90.0),
        ))
    return monitors


def build_shelf_detector(cam_config, shelf_cfg):
    slots_cfg = cam_config.get("shelf_slots", [])
    if not slots_cfg:
        return None, []
    slots = [
        ShelfSlot(
            aisle_id=s["aisle_id"], shelf_tier=s["shelf_tier"], slot_id=s["slot_id"],
            sku_id=s["sku_id"], rect=tuple(s["rect"]), expected_facings=s.get("expected_facings", 1),
            business_impact_multiplier=s.get("business_impact_multiplier", 1.0),
            baseline_velocity_per_hour=s.get("baseline_velocity_per_hour", 2.0),
            backroom_units=s.get("backroom_units", 0),
        )
        for s in slots_cfg
    ]
    return ShelfVoidDetector(slots, shelf_cfg), slots


def build_homography_source(cam_config):
    """Returns (static_homography, aruco_calibrator) -- exactly one of the
    two is non-None, or both are None if this camera has no `calibration`
    block configured. Static calibration is a fixed pixel<->cm mapping typed
    once; ArUco calibration is re-resolved every shelf-evaluation tick so a
    bumped camera "auto-heals" instead of silently reporting wrong cm gaps."""
    calibration = cam_config.get("calibration")
    if not calibration:
        return None, None
    mode = calibration.get("mode", "static")
    if mode == "aruco":
        markers = {int(k): tuple(v) for k, v in calibration["markers"].items()}
        return None, ArucoHomographyCalibrator(markers)
    return PlanarHomography(calibration["pixel_points"], calibration["metric_points_cm"]), None


def run_camera_pipeline(cam_name, cam_config, args, services: Services, stop_event):
    db, dispatcher = services.db, services.dispatcher
    inventory, pos, task_manager = services.inventory, services.pos, services.tasks

    source = cam_config.get("source", 0)
    override = getattr(args, f"{cam_name.replace('_cam', '')}_source", None)
    if override is not None:
        source = override

    try:
        frame_source = FrameSource(source, loop=args.loop)
    except RuntimeError as exc:
        logger.error("[%s] %s -- skipping this camera", cam_name, exc)
        return

    tracker = build_tracker(args.detector, CONFIG)
    footfall_counters = build_footfall_counters(cam_config, CONFIG.footfall)
    queue_monitors = build_queue_monitors(cam_config, CONFIG.queue)
    shelf_detector, _shelf_slots = build_shelf_detector(cam_config, CONFIG.shelf)
    static_homography, aruco_calibrator = build_homography_source(cam_config)
    label_reader = LabelReader() if (args.show and args.ocr and shelf_detector is not None) else None

    heatmap = None
    heatmap_enabled = bool(cam_config.get("heatmap"))
    last_heatmap_publish = 0.0

    last_shelf_eval = 0.0
    last_slot_states = []
    last_ocr_labels = {}
    last_queue_write = {}
    surge_alerted_until = 0.0

    logger.info("[%s] pipeline started (source=%s, detector=%s)", cam_name, source, args.detector)

    for frame in frame_source.frames():
        if stop_event.is_set():
            break

        tracks = tracker.update(frame)

        if heatmap_enabled:
            if heatmap is None:
                h, w = frame.shape[:2]
                heatmap = HeatmapAccumulator(width=w, height=h)
            heatmap.update(tracks)
            now_hm = time.time()
            if now_hm - last_heatmap_publish >= 2.0:
                last_heatmap_publish = now_hm
                LIVE_STATE.update_heatmap(
                    cam_name,
                    heatmap.render_png_bytes("traffic"),
                    heatmap.render_png_bytes("dwell"),
                )

        for counter in footfall_counters:
            events = counter.update(tracks)
            if events:
                for direction in events:
                    db.insert_footfall_event(direction)
                LIVE_STATE.update_footfall(counter.total_in, counter.total_out)

            surge = counter.surge_rate()
            if surge >= CONFIG.queue.surge_rate and time.time() > surge_alerted_until:
                dispatcher.dispatch(AlertEvent(
                    event_type="FOOTFALL_SURGE",
                    priority=PRIORITY_P2,
                    message=f"Inbound footfall up {surge * 100:.0f}% vs. the previous window at "
                            f"'{counter.tripwire.name}' -- consider opening additional lanes.",
                    payload={"camera": cam_name, "tripwire": counter.tripwire.name, "surge_rate": surge},
                ))
                surge_alerted_until = time.time() + CONFIG.queue.confirmation_window

        for monitor in queue_monitors:
            state = monitor.evaluate(tracks)
            LIVE_STATE.update_queue(state.lane_id, {
                "lane_id": state.lane_id,
                "people_count": state.people_count,
                "qualifying_shoppers": state.qualifying_shoppers,
                "estimated_wait_seconds": state.estimated_wait_seconds,
                "alert_triggered": state.alert_triggered,
                "threshold": CONFIG.queue.threshold,
                "arrival_rate_per_minute": state.arrival_rate_per_minute,
                "service_rate_per_minute": state.service_rate_per_minute,
                "congestion_index": state.congestion_index,
                "predicted_queue_depth": state.predicted_queue_depth,
                "recommendation": state.recommendation,
            })

            now = time.time()
            if now - last_queue_write.get(state.lane_id, 0) >= 5.0:
                db.insert_queue_telemetry(state.lane_id, state.people_count, state.estimated_wait_seconds)
                last_queue_write[state.lane_id] = now

            was_alerting = getattr(monitor, "_was_alerting", False)
            if state.alert_triggered and not was_alerting:
                dispatcher.dispatch(AlertEvent(
                    event_type="CHECKOUT_QUEUE_SURGE",
                    priority=PRIORITY_P1,
                    message=f"Checkout congested: {state.qualifying_shoppers} customers waiting >= "
                            f"{int(CONFIG.queue.dwell_time_seconds)}s at {state.lane_id}. Open a backup register.",
                    payload={"camera": cam_name, "lane_id": state.lane_id,
                             "people_count": state.people_count,
                             "estimated_wait_seconds": state.estimated_wait_seconds},
                ))
            monitor._was_alerting = state.alert_triggered

            # Congestion Index forecast: fire only on a recommendation-tier transition
            # so a sustained state doesn't spam the same alert every frame.
            last_recommendation = getattr(monitor, "_last_recommendation", None)
            if state.recommendation != last_recommendation:
                if state.recommendation == RECOMMENDATION_CRITICAL:
                    dispatcher.dispatch(AlertEvent(
                        event_type="QUEUE_CONGESTION_FORECAST",
                        priority=PRIORITY_P1,
                        message=f"{state.lane_id}: congestion index {state.congestion_index:.2f}, "
                                f"predicted depth {state.predicted_queue_depth:.1f} within "
                                f"{int(CONFIG.queue.forecast_horizon_seconds / 60)} min -- open additional lanes now.",
                        payload={"camera": cam_name, "lane_id": state.lane_id,
                                 "congestion_index": state.congestion_index,
                                 "predicted_queue_depth": state.predicted_queue_depth,
                                 "arrival_rate_per_minute": state.arrival_rate_per_minute},
                    ))
                    lane_priority = PriorityResult(
                        score=90, bucket="P1", raw=0.0, V=state.arrival_rate_per_minute,
                        G=0.0, A=1.0, D=0.0, cause=RECOMMENDATION_CRITICAL,
                    )
                    task = task_manager.create_task(
                        kind="lane", zone="CHECKOUT",
                        title=f"Open another counter -- {state.lane_id}",
                        reason=f"CI={state.congestion_index:.2f}, predicted depth "
                               f"{state.predicted_queue_depth:.1f} in {int(CONFIG.queue.forecast_horizon_seconds / 60)} min",
                        cause=RECOMMENDATION_CRITICAL, priority=lane_priority, lane_id=state.lane_id,
                    )
                    logger.info("[%s] created task %s (open lane, %s)", cam_name, task.task_id, state.lane_id)
                elif state.recommendation == RECOMMENDATION_WARNING:
                    dispatcher.dispatch(AlertEvent(
                        event_type="QUEUE_CONGESTION_WARNING",
                        priority=PRIORITY_P2,
                        message=f"{state.lane_id}: approaching capacity (CI={state.congestion_index:.2f}) "
                                f"within {int(CONFIG.queue.forecast_horizon_seconds / 60)} min.",
                        payload={"camera": cam_name, "lane_id": state.lane_id,
                                 "congestion_index": state.congestion_index},
                    ))
                elif state.recommendation == RECOMMENDATION_IDLE:
                    dispatcher.dispatch(AlertEvent(
                        event_type="QUEUE_EXCESS_CAPACITY",
                        priority=PRIORITY_P4,
                        message=f"{state.lane_id}: sustained low demand (CI={state.congestion_index:.2f}) "
                                f"-- consider closing this lane and reallocating staff.",
                        payload={"camera": cam_name, "lane_id": state.lane_id,
                                 "congestion_index": state.congestion_index},
                    ))
                monitor._last_recommendation = state.recommendation

        if shelf_detector is not None:
            now = time.time()
            if now - last_shelf_eval >= CONFIG.shelf.evaluation_interval_seconds:
                last_shelf_eval = now
                person_bboxes = [t.bbox for t in tracks]

                current_homography = static_homography
                if aruco_calibrator is not None:
                    current_homography = aruco_calibrator.try_recalibrate(frame)

                slot_states = shelf_detector.evaluate(frame, person_bboxes, homography=current_homography)
                last_slot_states = slot_states

                if label_reader is not None:
                    last_ocr_labels = {
                        s.slot.slot_id: label_reader.read(frame, s.slot.rect) for s in slot_states
                    }

                for state in slot_states:
                    LIVE_STATE.update_shelf_slot(state.slot.slot_id, {
                        "slot_id": state.slot.slot_id,
                        "sku_id": state.slot.sku_id,
                        "aisle_id": state.slot.aisle_id,
                        "shelf_tier": state.slot.shelf_tier,
                        "void_ratio": state.void_ratio,
                        "fill_ratio": state.fill_ratio,
                        "gap_cm": state.gap_cm,
                        "low_stock_warning": state.low_stock_warning,
                        "out_of_stock_alert": state.out_of_stock_alert,
                        "occluded": state.occluded,
                    })
                    if state.occluded:
                        continue
                    db.insert_shelf_event(
                        state.slot.aisle_id, state.slot.shelf_tier, state.slot.slot_id,
                        state.slot.sku_id, state.void_ratio, state.out_of_stock_alert,
                    )
                    # Tracked per severity level (not a single "was_alerting" flag) so
                    # that a slot which already fired LOW_STOCK can still escalate to
                    # SHELF_OUT_OF_STOCK later -- a single shared flag would let the
                    # earlier, lower-severity alert permanently suppress the later,
                    # more urgent one for the rest of the session.
                    last_level = getattr(state.slot, "_last_alert_level", "none")
                    current_level = (
                        "out_of_stock" if state.out_of_stock_alert
                        else "low_stock" if state.low_stock_warning
                        else "none"
                    )
                    if current_level == "out_of_stock" and last_level != "out_of_stock":
                        dispatcher.dispatch(AlertEvent(
                            event_type="SHELF_OUT_OF_STOCK",
                            priority=PRIORITY_P1,
                            message=f"Restock needed: {state.slot.slot_id} ({state.slot.sku_id}) "
                                    f"void ratio {state.void_ratio * 100:.0f}%.",
                            payload={"camera": cam_name, "slot_id": state.slot.slot_id,
                                     "sku_id": state.slot.sku_id, "void_ratio": state.void_ratio},
                        ))

                        # Fusion: combine the visual void with mock POS/inventory signals
                        # to decide *why* the shelf is empty and how urgent it is, then
                        # open (and assign) a task -- or, for PURCHASING_ALERT, skip the
                        # floor task entirely since restocking wouldn't help.
                        slot = state.slot
                        pos_available = pos.is_available()
                        velocity_class = pos.velocity_class(slot.sku_id, slot.baseline_velocity_per_hour)
                        velocity_per_hour = pos.velocity_per_hour(slot.sku_id)
                        backroom_units = inventory.get_backroom_units(slot.sku_id)
                        cause = fusion.classify_cause(pos_available, velocity_class, backroom_units)
                        gap_elapsed_seconds = (
                            state.consecutive_alert_frames * CONFIG.shelf.evaluation_interval_seconds
                        )
                        priority = fusion.compute_priority(
                            velocity_per_hour=velocity_per_hour, baseline_per_hour=slot.baseline_velocity_per_hour,
                            business_impact_multiplier=slot.business_impact_multiplier, void_ratio=state.void_ratio,
                            gap_elapsed_seconds=gap_elapsed_seconds, backroom_units=backroom_units, cause=cause,
                        )

                        if cause == fusion.CAUSE_PURCHASING:
                            dispatcher.dispatch(AlertEvent(
                                event_type="PURCHASING_ALERT", priority=PRIORITY_P2,
                                message=f"Reorder needed: {slot.sku_id} -- shelf and backroom both empty "
                                        f"(velocity {velocity_class}). No floor task created.",
                                payload={"camera": cam_name, "slot_id": slot.slot_id, "sku_id": slot.sku_id,
                                         "cause": cause, "priority_score": priority.score},
                            ))
                        else:
                            kind = {
                                fusion.CAUSE_REPLENISHMENT: "restock",
                                fusion.CAUSE_LOW_PRIORITY: "audit",
                                fusion.CAUSE_VISUAL_ONLY: "check",
                            }[cause]
                            task = task_manager.create_task(
                                kind=kind, zone=slot.aisle_id,
                                title=f"{kind.capitalize()} {slot.sku_id} at {slot.slot_id}",
                                reason=f"{cause}: void {state.void_ratio:.2f}, backroom {backroom_units} units, "
                                       f"velocity {velocity_class}",
                                cause=cause, priority=priority, slot_id=slot.slot_id, sku_id=slot.sku_id,
                            )
                            logger.info("[%s] created task %s (%s, %s %d) for %s",
                                        cam_name, task.task_id, cause, priority.bucket, priority.score, slot.slot_id)
                    elif current_level == "low_stock" and last_level == "none":
                        dispatcher.dispatch(AlertEvent(
                            event_type="SHELF_LOW_STOCK",
                            priority=PRIORITY_P2,
                            message=f"Low stock: {state.slot.slot_id} ({state.slot.sku_id}) "
                                    f"fill ratio {state.fill_ratio * 100:.0f}%.",
                            payload={"camera": cam_name, "slot_id": state.slot.slot_id,
                                     "sku_id": state.slot.sku_id, "fill_ratio": state.fill_ratio},
                        ))
                    state.slot._last_alert_level = current_level

        if args.show:
            display = frame.copy()
            draw_tripwires(display, [c.tripwire for c in footfall_counters])
            draw_polygons(display, [m.polygon for m in queue_monitors])
            if shelf_detector is not None:
                draw_shelf_slots(display, last_slot_states)
                if label_reader is not None:
                    for state in last_slot_states:
                        reading = last_ocr_labels.get(state.slot.slot_id)
                        if reading is None:
                            continue
                        x, y, w, h = state.slot.rect
                        cv2.putText(display, f"OCR {reading.confidence:.2f}: {reading.text[:24]}",
                                    (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            draw_tracks(display, tracks)
            cv2.imshow(cam_name, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

    frame_source.release()
    if args.show:
        cv2.destroyWindow(cam_name)
    logger.info("[%s] pipeline stopped", cam_name)


def build_argparser():
    parser = argparse.ArgumentParser(description="Edge Retail Intelligence Platform")
    parser.add_argument("--layout", default=CONFIG.layout_path, help="Path to store_layout.json")
    parser.add_argument("--only", default=None, help="Comma-separated camera keys to run (default: all in layout)")
    parser.add_argument("--detector", choices=["motion", "yolo"], default=CONFIG.tracker.backend)
    parser.add_argument("--entrance-source", default=None, help="Override entrance_cam source")
    parser.add_argument("--checkout-source", default=None, help="Override checkout_cam source")
    parser.add_argument("--shelf-source", default=None, help="Override shelf_cam source")
    parser.add_argument("--loop", action="store_true", help="Loop demo video files at EOF instead of stopping")
    parser.add_argument("--show", action="store_true", help="Open local debug overlay window(s)")
    parser.add_argument("--ocr", action="store_true",
                         help="Overlay shelf-tag OCR text in --show mode (needs pytesseract + Tesseract binary)")
    parser.add_argument("--no-server", action="store_true", help="Run the vision pipeline only; skip the API/dashboard")
    parser.add_argument("--host", default=CONFIG.api_host)
    parser.add_argument("--port", type=int, default=CONFIG.api_port)
    return parser


def main():
    args = build_argparser().parse_args()
    layout = load_layout(args.layout)
    cameras = layout.get("cameras", {})

    if args.only:
        requested = set(args.only.split(","))
        cameras = {k: v for k, v in cameras.items() if k in requested}
        if not cameras:
            logger.error("No cameras matched --only=%s (available: %s)", args.only, list(layout.get("cameras", {}).keys()))
            return

    db = Database(CONFIG.db_path)
    dispatcher = AlertDispatcher(CONFIG.alerts)
    roster = build_roster(layout)
    inventory = build_inventory_adapter(layout.get("cameras", {}))
    pos = MockPOSAdapter()
    task_manager = TaskManager(roster, db=db)
    services = Services(db=db, dispatcher=dispatcher, inventory=inventory, pos=pos, tasks=task_manager)

    stop_event = threading.Event()
    threads = []
    for cam_name, cam_config in cameras.items():
        t = threading.Thread(
            target=run_camera_pipeline,
            args=(cam_name, cam_config, args, services, stop_event),
            daemon=True,
        )
        t.start()
        threads.append(t)

    if not threads:
        logger.error("No camera pipelines started.")
        return

    ticker = threading.Thread(target=run_task_ticker, args=(task_manager, stop_event), daemon=True)
    ticker.start()
    threads.append(ticker)

    if args.no_server:
        try:
            while any(t.is_alive() for t in threads):
                time.sleep(1)
        except KeyboardInterrupt:
            stop_event.set()
        return

    app = create_app(db, dispatcher, task_manager, pos)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=5)


if __name__ == "__main__":
    main()
