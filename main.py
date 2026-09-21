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
    python main.py --no-server                     # run the vision pipeline only, no dashboard/API
"""
import argparse
import json
import logging
import threading
import time

import cv2
import uvicorn

from app.alerts.dispatcher import AlertDispatcher, AlertEvent, PRIORITY_P1, PRIORITY_P2
from app.database import Database
from app.geometry import Polygon, Tripwire
from app.server.api import create_app
from app.state import LIVE_STATE
from app.vision.footfall import FootfallCounter
from app.vision.overlay import draw_polygons, draw_shelf_slots, draw_tracks, draw_tripwires
from app.vision.queue_monitor import QueueMonitor
from app.vision.shelf_monitor import ShelfSlot, ShelfVoidDetector
from app.vision.stream import FrameSource
from app.vision.tracker import build_tracker
from config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")


def load_layout(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        return None
    slots = [
        ShelfSlot(
            aisle_id=s["aisle_id"], shelf_tier=s["shelf_tier"], slot_id=s["slot_id"],
            sku_id=s["sku_id"], rect=tuple(s["rect"]), expected_facings=s.get("expected_facings", 1),
        )
        for s in slots_cfg
    ]
    return ShelfVoidDetector(slots, shelf_cfg)


def run_camera_pipeline(cam_name, cam_config, args, db, dispatcher, stop_event):
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
    shelf_detector = build_shelf_detector(cam_config, CONFIG.shelf)

    last_shelf_eval = 0.0
    last_slot_states = []
    last_queue_write = {}
    surge_alerted_until = 0.0

    logger.info("[%s] pipeline started (source=%s, detector=%s)", cam_name, source, args.detector)

    for frame in frame_source.frames():
        if stop_event.is_set():
            break

        tracks = tracker.update(frame)

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

        if shelf_detector is not None:
            now = time.time()
            if now - last_shelf_eval >= CONFIG.shelf.evaluation_interval_seconds:
                last_shelf_eval = now
                person_bboxes = [t.bbox for t in tracks]
                slot_states = shelf_detector.evaluate(frame, person_bboxes)
                last_slot_states = slot_states
                for state in slot_states:
                    LIVE_STATE.update_shelf_slot(state.slot.slot_id, {
                        "slot_id": state.slot.slot_id,
                        "sku_id": state.slot.sku_id,
                        "aisle_id": state.slot.aisle_id,
                        "shelf_tier": state.slot.shelf_tier,
                        "void_ratio": state.void_ratio,
                        "fill_ratio": state.fill_ratio,
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
                    was_alerting = getattr(state.slot, "_was_alerting", False)
                    if state.out_of_stock_alert and not was_alerting:
                        dispatcher.dispatch(AlertEvent(
                            event_type="SHELF_OUT_OF_STOCK",
                            priority=PRIORITY_P1,
                            message=f"Restock needed: {state.slot.slot_id} ({state.slot.sku_id}) "
                                    f"void ratio {state.void_ratio * 100:.0f}%.",
                            payload={"camera": cam_name, "slot_id": state.slot.slot_id,
                                     "sku_id": state.slot.sku_id, "void_ratio": state.void_ratio},
                        ))
                    elif state.low_stock_warning and not state.out_of_stock_alert and not was_alerting:
                        dispatcher.dispatch(AlertEvent(
                            event_type="SHELF_LOW_STOCK",
                            priority=PRIORITY_P2,
                            message=f"Low stock: {state.slot.slot_id} ({state.slot.sku_id}) "
                                    f"fill ratio {state.fill_ratio * 100:.0f}%.",
                            payload={"camera": cam_name, "slot_id": state.slot.slot_id,
                                     "sku_id": state.slot.sku_id, "fill_ratio": state.fill_ratio},
                        ))
                    state.slot._was_alerting = state.out_of_stock_alert or state.low_stock_warning

        if args.show:
            display = frame.copy()
            draw_tripwires(display, [c.tripwire for c in footfall_counters])
            draw_polygons(display, [m.polygon for m in queue_monitors])
            if shelf_detector is not None:
                draw_shelf_slots(display, last_slot_states)
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

    stop_event = threading.Event()
    threads = []
    for cam_name, cam_config in cameras.items():
        t = threading.Thread(
            target=run_camera_pipeline,
            args=(cam_name, cam_config, args, db, dispatcher, stop_event),
            daemon=True,
        )
        t.start()
        threads.append(t)

    if not threads:
        logger.error("No camera pipelines started.")
        return

    if args.no_server:
        try:
            while any(t.is_alive() for t in threads):
                time.sleep(1)
        except KeyboardInterrupt:
            stop_event.set()
        return

    app = create_app(db, dispatcher)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=5)


if __name__ == "__main__":
    main()
