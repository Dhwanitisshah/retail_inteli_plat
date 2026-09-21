# Edge Retail Intelligence Platform (MVP)

A working implementation of the MVP slice from the platform spec: **entrance
footfall tracking** + **checkout queue alerting**, plus a class-agnostic
**shelf void detector**, running entirely on-device with zero PII, an
embedded SQLite store, and a live FastAPI + WebSocket dashboard.

It runs on a laptop out of the box (no GPU, no ML weights) using a
lightweight OpenCV motion-blob tracker, with a pluggable real YOLOv8n +
ByteTrack backend for production edge hardware.

## Architecture at a glance

```
FrameSource (webcam / video file / RTSP)
        │
        ▼
PersonTracker  ── MotionBlobTracker (default) or YoloByteTrackTracker (--detector yolo)
        │  (ephemeral track_id + ground-contact centroid only -- no Re-ID, no faces)
        ├──► FootfallCounter   -- tripwire crossing (Section 3.1)
        ├──► QueueMonitor      -- dwell-time + confirmation-window alerting (Section 3.2/4.1)
        └──► ShelfVoidDetector -- per-slot void/fill ratio, occlusion-gated (Section 3.3)
                │
                ▼
        SQLite (footfall_events / queue_telemetry / shelf_compliance_events)
                │
                ▼
        AlertDispatcher ──► console log, Telegram bot, generic webhook, dashboard WebSocket
                │
                ▼
        FastAPI + static dashboard (REST + /ws/live)
```

Each named camera in `sample_config/store_layout.json` runs its own
ingestion + tracking pipeline on a background thread; `main.py` wires them
all to one shared database, alert dispatcher, and dashboard.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Generate short synthetic demo clips (no camera needed)
python scripts/generate_demo_clips.py

# Run the entrance footfall pipeline against the demo clip + dashboard
python main.py --only entrance_cam --entrance-source data/demo_clips/entrance.mp4
```

Then open **http://localhost:8000** for the live dashboard.

To use a real webcam instead: `python main.py --only entrance_cam
--entrance-source 0 --show` (`--show` opens a local OpenCV overlay window
with the tripwire/ROI/track drawings for debugging).

Run all three camera pipelines at once only if you have three separate
video sources (three files, or three distinct cameras) -- most OS camera
backends won't let you open the same physical webcam twice:

```bash
python main.py \
  --entrance-source data/demo_clips/entrance.mp4 \
  --checkout-source data/demo_clips/checkout.mp4 \
  --shelf-source data/demo_clips/shelf.mp4
```

### Verifying it without any video source

```bash
python tests/test_pipeline_smoke.py
```

Runs the tripwire crossing, queue debounce state machine, shelf void
detection + occlusion gating, SQLite persistence, and alert fan-out against
synthetic data with zeroed timing thresholds -- no camera, no wall-clock
waiting.

## What's spec-accurate vs. a stand-in

| Component | This implementation | Spec's production target |
|---|---|---|
| Tripwire footfall counting | Exact vector-crossing math from Section 3.1 | Same |
| Queue dwell/threshold/confirmation logic | Exact state machine from Section 4.1's pseudocode | Same |
| Wait time estimate | Spec's `(people * avg_service_time) / active_cashiers` formula | Same |
| Person detection/tracking | `MotionBlobTracker` (background subtraction, default) **or** `YoloByteTrackTracker` (`--detector yolo`, needs `pip install -r requirements-yolo.txt`) | YOLOv8n/YOLOv10n + ByteTrack on TensorRT |
| Shelf void detection | Classical CV heuristic: per-slot Canny edge-density vs. a calibrated "stocked" reference, plus row-activity for vertical fill ratio | Trained YOLOv8-Seg instance segmentation (`product_cluster` / `empty_shelf_void`) |
| Planogram slot mapping | Slots are pre-calibrated pixel rectangles in `store_layout.json` (no live homography) | Pixel→metric homography + planogram DB lookup |
| Alert channels | Console log + optional Telegram bot + optional generic webhook + dashboard WebSocket | Zebra handhelds, PA chimes, POS pop-ups |
| Persistence | SQLite, schema identical to the spec (Section 5.1) | Same |
| Edge-to-cloud sync | Not implemented (single-store, local-only) | Store-and-forward batched JSON over MQTT/HTTPS |

The shelf detector's contract (`ShelfSlot` in, `ShelfSlotState` out with
`void_ratio`/`fill_ratio`/`out_of_stock_alert`) is intentionally identical to
what a real segmentation model would need to produce, so swapping in a
trained YOLOv8-Seg model later only touches `app/vision/shelf_monitor.py`.

## Configuration

Thresholds live in `config.py` and mirror the spec's numbers exactly
(`QUEUE_THRESHOLD=4`, `DWELL_TIME_SECONDS=60`, `CONFIRMATION_WINDOW=120`,
`void_ratio_alert=0.65`, `fill_ratio_warn=0.30`). Override via environment
variables (`STORE_ID`, `TRACKER_BACKEND`, `TELEGRAM_BOT_TOKEN`,
`TELEGRAM_CHAT_ID`, `ALERT_WEBHOOK_URL`, etc. -- see `config.py`) or edit the
dataclass defaults directly.

Camera geometry (tripwires, queue ROI polygons, shelf slot rectangles) lives
in `sample_config/store_layout.json`. Coordinates are pixel coordinates in
that camera's frame; recalibrate them for your own camera placement by
drawing over a captured frame (`--show` overlays the currently configured
geometry so you can iterate).

## Project structure

```
edge-retail-ai/
├── main.py                     # orchestrator: wires cameras -> pipeline -> db/alerts/dashboard
├── config.py                   # thresholds + runtime settings (dataclasses)
├── app/
│   ├── geometry.py              # Tripwire crossing math, Polygon point-in-polygon
│   ├── database.py              # SQLite schema + queries (matches spec Section 5.1)
│   ├── state.py                 # thread-safe live snapshot shared with the API layer
│   ├── vision/
│   │   ├── stream.py             # webcam/file/RTSP frame source
│   │   ├── tracker.py            # MotionBlobTracker + YoloByteTrackTracker
│   │   ├── footfall.py           # tripwire IN/OUT counting + surge-rate calc
│   │   ├── queue_monitor.py      # dwell-time + confirmation-window alerting
│   │   ├── shelf_monitor.py      # void/fill ratio + human-occlusion gating
│   │   └── overlay.py            # --show debug drawing helpers
│   ├── alerts/dispatcher.py     # in-process pub/sub + Telegram/webhook fan-out
│   └── server/
│       ├── api.py                # FastAPI REST + WebSocket
│       └── static/dashboard.html # single-page live dashboard
├── sample_config/store_layout.json  # camera/tripwire/ROI/shelf-slot geometry
├── scripts/generate_demo_clips.py   # synthetic demo videos, no camera needed
└── tests/test_pipeline_smoke.py     # end-to-end logic tests, no pytest required
```

## Known limitations

- The motion-blob tracker is a wiring/demo aid, not a detection-accuracy
  benchmark -- it will false-positive on lighting changes and can't
  distinguish people from other moving objects. Use `--detector yolo` for
  anything beyond a logic demo.
- The shelf void heuristic is calibrated per-slot from whatever frame it
  first sees, so point the shelf camera at a genuinely fully-stocked shelf
  before starting a session (or call `ShelfVoidDetector.calibrate()`
  explicitly with a known-good reference frame).
- Single-store, local-only: there's no store-and-forward sync to a central
  cloud warehouse yet (Section 8 of the spec).
