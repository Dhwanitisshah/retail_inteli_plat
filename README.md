# SAMGRAHA — Edge Retail Intelligence Platform

**Team AUREX · Smart India Hackathon 2026 · Problem Statement 26179 (Hardware / Miscellaneous)**

📺 **[Illustrated walkthrough (live demo page)](https://claude.ai/artifact/FRPQemet67X3Tigs88vQ6J)** — all 5
pipeline modules paired with real store/checkout/backroom photos, including
a heatmap that's genuine code output, not a mockup. *(Private Claude
artifact link — open it and use its Share menu if it asks for access.)*

An edge-native computer vision platform that runs entirely on in-store
hardware: **entrance footfall tracking**, **checkout queue alerting with a
predictive Congestion Index**, and **class-agnostic shelf void detection**
with metric (cm) gap measurement -- all with zero PII, an embedded SQLite
store, and a live FastAPI + WebSocket dashboard.

It runs on a laptop out of the box (no GPU, no ML weights) using a
lightweight OpenCV motion-blob tracker, with a pluggable real YOLOv8n +
ByteTrack backend for production edge hardware. See
[HARDWARE.md](HARDWARE.md) for the physical deployment plan (cameras, edge
compute tiers, network topology, UPS/power) -- this repo is the software
that runs on it. See [FEATURE_COMPARISON.md](FEATURE_COMPARISON.md) for a
subsystem-by-subsystem comparison against "The Physical Ledger" reference
demo, including what's still not implemented and why.

## Architecture at a glance

```
FrameSource (webcam / video file / RTSP)
        │
        ▼
PersonTracker  ── MotionBlobTracker (default) or YoloByteTrackTracker (--detector yolo)
        │  (ephemeral track_id + ground-contact centroid only -- no Re-ID, no faces)
        ├──► FootfallCounter   -- tripwire crossing + surge-rate detection
        ├──► HeatmapAccumulator -- traffic/dwell 2D Gaussian floor heatmap
        ├──► QueueMonitor      -- dwell-threshold alert + Congestion Index forecast (CI = λ/(c·μ))
        │        │ (critical forecast)
        └──► ShelfVoidDetector -- per-slot void/fill ratio, occlusion-gated, metric gap (cm)
                 │ (out-of-stock)               ▲
                 ▼                    PlanarHomography / ArucoHomographyCalibrator
        Fusion (app/fusion.py)         (pixel -> cm, auto-heals against camera drift)
        shelf void × MockInventoryAdapter × MockPOSAdapter
                 │ (REPLENISHMENT_REQUIRED / PURCHASING_ALERT / LOW_PRIORITY_OOS / VISUAL_ALERT_ONLY)
                 ▼
        TaskManager (app/tasks.py) ── assigns via Roster, tracks SLA,
                 │                    re-verifies against the live camera signal
                 ▼
        SQLite (footfall_events / queue_telemetry / shelf_compliance_events / tasks)
                │
                ▼
        AlertDispatcher ──► console log, Telegram bot, generic webhook, dashboard WebSocket
                │
                ▼
        FastAPI + static dashboard (REST + /ws/live + /api/heatmap/*.png + /api/tasks)
```

Each named camera in `sample_config/store_layout.json` runs its own
ingestion + tracking pipeline on a background thread; `main.py` wires them
all to one shared database, alert dispatcher, task engine, and dashboard.
A single dedicated ticker thread runs `TaskManager.tick()` so verification
and SLA checks never race with the per-camera pipelines.

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
with the tripwire/ROI/track drawings for debugging; add `--ocr` to also
overlay shelf-tag OCR text, if you've installed
`requirements-ocr.txt` + the Tesseract binary).

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

20 tests covering: tripwire crossing, the queue dwell/confirmation state
machine, the Congestion Index formula and its recommendation tiers, planar
homography pixel→cm projection, shelf void detection + occlusion gating +
metric gap measurement, the heatmap accumulator, OCR's graceful fallback,
the mock POS/inventory adapters, the fusion cause-classification truth
table and bounded priority formula, roster assignment scoring, the full
task lifecycle (created → assigned → verified-resolved, and separately
escalated → VERIFY_UNAVAILABLE), SLA-breach tracking, SQLite persistence,
and alert fan-out -- all against synthetic data with zeroed timing
thresholds, no camera or wall-clock waiting required.

## Task & fusion engine

The MVP could detect and alert; it had no idea whether anyone fixed
anything. When a shelf slot crosses into `out_of_stock_alert`, `main.py`
now runs it through `app/fusion.py` against two **mock** adapters
(`app/pos.py`, `app/inventory.py` -- in-memory stand-ins behind the same
`ABC` a real POS/WMS integration would implement) to classify *why* it's
empty and *how urgent* that is, then opens a `Task` via `app/tasks.py`,
assigned to a rostered staff member (`sample_config/store_layout.json`'s
`"roster"` block). A queue's Congestion Index forecast hitting the critical
tier opens a "lane" task the same way. Marking a task done never resolves
it by itself -- a dedicated ticker thread re-checks the *live* camera/queue
signal on a timer before closing it, exactly like the checkout-queue
recommendation already re-checks itself.

Poke it from the dashboard's **Tasks** panel, or directly:

```bash
curl http://localhost:8000/api/tasks                         # list tasks
curl -X POST http://localhost:8000/api/tasks/T-0001/complete  # mark done -> awaits camera re-check
curl -X POST http://localhost:8000/api/pos/sale \
  -H "Content-Type: application/json" -d '{"sku_id": "SKU_8901030", "units": 5}'  # feed the velocity classifier
curl -X POST http://localhost:8000/api/pos/availability \
  -H "Content-Type: application/json" -d '{"available": false}'  # simulate a POS outage -> VISUAL_ALERT_ONLY
```

## What's spec-accurate vs. a stand-in

| Component | This implementation | Production target |
|---|---|---|
| Tripwire footfall counting | Exact vector-crossing math | Same |
| Queue dwell/threshold/confirmation alert | Exact state machine from the MVP pseudocode | Same |
| Congestion Index forecast | `CI = λ/(c·μ)` computed from *observed* arrival rate; recommendation tiers (optimal/warning/critical/idle) match the spec's thresholds | Spec's full version feed-forwards λ from upstream "final-stage" zone dwell, which needs extra camera coverage not in this layout |
| Wait time estimate | `(people * avg_service_time) / active_cashiers` | Same |
| Person detection/tracking | `MotionBlobTracker` (background subtraction, default) **or** `YoloByteTrackTracker` (`--detector yolo`, needs `pip install -r requirements-yolo.txt`) | YOLOv8n/v10n + ByteTrack on TensorRT |
| Shelf void detection | Classical CV heuristic: per-slot Canny edge-density vs. a calibrated "stocked" reference, plus row/column activity for fill ratio and gap extent | Trained YOLOv8-Seg instance segmentation (`product_cluster` / `empty_shelf_void`) |
| Metric shelf gap (cm) | `PlanarHomography` (static 4-point calibration) or `ArucoHomographyCalibrator` (auto-heals from 4 ArUco markers every eval tick) | Same `x' = Hx` planar projective geometry |
| Movement heatmaps | `HeatmapAccumulator`: decaying 2D Gaussian splat grid, traffic + dwell layers, served as PNG | Same, published as periodic normalized matrices |
| Shelf-tag OCR | Optional (`--ocr`, needs `pytesseract` + Tesseract binary); pure demo overlay, never load-bearing for alerting | Not actually in the spec's alerting logic either -- SKU identity comes from the planogram slot, not the printed tag |
| Alert channels | Console log + optional Telegram bot + optional generic webhook + dashboard WebSocket | Zebra handhelds, PA chimes, POS pop-ups (see HARDWARE.md) |
| Persistence | SQLite, WAL mode (survives a hard power cut mid-write) | Same, backed by a UPS for graceful shutdown |
| Edge-to-cloud sync | Not implemented (single-store, local-only) | Store-and-forward batched JSON over MQTT/HTTPS |
| Thermal/UPS monitoring, PoE, rack hardware | Not implementable without physical hardware -- see [HARDWARE.md](HARDWARE.md) | Real sensors/hardware |
| Cause classification + priority scoring + task lifecycle | Real: `app/fusion.py` + `app/tasks.py`, against **mock** POS/inventory adapters (`app/pos.py`, `app/inventory.py`) | Same logic against a real POS/WMS integration -- see [FEATURE_COMPARISON.md](FEATURE_COMPARISON.md) |

The shelf detector's contract (`ShelfSlot` in, `ShelfSlotState` out with
`void_ratio`/`fill_ratio`/`gap_cm`/`out_of_stock_alert`) is intentionally
identical to what a real segmentation model would need to produce, so
swapping in a trained YOLOv8-Seg model later only touches
`app/vision/shelf_monitor.py`.

## Configuration

Thresholds live in `config.py`:

- Queue: `threshold=4`, `dwell_time_seconds=60`, `confirmation_window=120`,
  plus the Congestion Index engine's `ci_warning_threshold=0.70`,
  `ci_critical_threshold=1.00`, `ci_idle_threshold=0.35`,
  `ci_idle_sustained_seconds=600`, `arrival_window_seconds=300`,
  `forecast_horizon_seconds=300`.
- Shelf: `void_ratio_alert=0.65`, `fill_ratio_warn=0.30`,
  `consecutive_frames_required=3`, `evaluation_interval_seconds=5.0`.

Override via environment variables (`STORE_ID`, `TRACKER_BACKEND`,
`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `ALERT_WEBHOOK_URL`, etc. -- see
`config.py`) or edit the dataclass defaults directly.

Camera geometry lives in `sample_config/store_layout.json`: tripwires,
queue ROI polygons, shelf slot rectangles, an optional `"heatmap": true`
flag per camera, and an optional `"calibration"` block for metric shelf
gaps:

```json
"calibration": {
  "mode": "static",
  "pixel_points": [[0, 100], [630, 100], [630, 250], [0, 250]],
  "metric_points_cm": [[0, 0], [180, 0], [180, 45], [0, 45]]
}
```

or, to auto-heal against camera drift instead of trusting a one-time static
calibration (see HARDWARE.md):

```json
"calibration": {
  "mode": "aruco",
  "markers": { "0": [0, 0], "1": [180, 0], "2": [180, 45], "3": [0, 45] }
}
```

`--show` overlays the currently configured geometry (tripwires, ROIs, shelf
slots with live void_ratio/gap_cm) so you can iterate on calibration.

## Project structure

```
edge-retail-ai/
├── main.py                     # orchestrator: wires cameras -> pipeline -> fusion/tasks -> db/alerts/dashboard
├── config.py                   # thresholds + runtime settings (dataclasses)
├── HARDWARE.md                 # physical deployment reference (cameras, edge tiers, network, UPS)
├── FEATURE_COMPARISON.md       # subsystem-by-subsystem comparison vs. the reference demo
├── app/
│   ├── geometry.py              # Tripwire crossing math, Polygon point-in-polygon
│   ├── database.py              # SQLite (WAL) schema + queries, incl. the tasks table
│   ├── state.py                 # thread-safe live snapshot shared with the API layer
│   ├── fusion.py                # cause classification + bounded priority scoring
│   ├── tasks.py                 # Task state machine + verification/SLA scheduler
│   ├── roster.py                # Staff + zone/workload assignment scoring
│   ├── pos.py                   # POSAdapter (ABC) + MockPOSAdapter (velocity classification)
│   ├── inventory.py              # InventoryAdapter (ABC) + MockInventoryAdapter (backroom/system counts)
│   ├── vision/
│   │   ├── stream.py             # webcam/file/RTSP frame source
│   │   ├── tracker.py            # MotionBlobTracker + YoloByteTrackTracker
│   │   ├── footfall.py           # tripwire IN/OUT counting + surge-rate calc
│   │   ├── queue_monitor.py      # dwell-threshold alert + Congestion Index forecast engine
│   │   ├── shelf_monitor.py      # void/fill ratio, occlusion gating, metric gap via homography
│   │   ├── homography.py         # PlanarHomography + ArucoHomographyCalibrator (x' = Hx)
│   │   ├── heatmap.py            # decaying 2D Gaussian traffic/dwell accumulator
│   │   ├── ocr.py                # optional pytesseract shelf-tag reader (graceful no-op fallback)
│   │   └── overlay.py            # --show debug drawing helpers
│   ├── alerts/dispatcher.py     # in-process pub/sub + Telegram/webhook fan-out
│   └── server/
│       ├── api.py                # FastAPI REST + WebSocket + heatmap PNG + task/POS endpoints
│       └── static/dashboard.html # single-page live dashboard (incl. the Tasks panel)
├── sample_config/store_layout.json  # camera/tripwire/ROI/shelf-slot/calibration/roster config
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
- The Congestion Index's λ_predicted is the *observed* arrival rate, not a
  feed-forward estimate from upstream store zones (that needs additional
  camera coverage this layout doesn't have).
- Single-store, local-only: there's no store-and-forward sync to a central
  cloud warehouse yet.
- OCR, thermal/UPS monitoring, and the associate-handheld/PA-chime alert
  channels are either optional/best-effort or entirely out of scope without
  physical hardware -- see [HARDWARE.md](HARDWARE.md).
- `app/pos.py` and `app/inventory.py` are in-memory mocks -- POS velocity
  starts at zero until something calls `/api/pos/sale`, and backroom counts
  only ever go down (nothing restocks them automatically). Point a real
  POS/WMS integration at the `POSAdapter`/`InventoryAdapter` interfaces to
  replace them without touching `app/fusion.py` or `app/tasks.py`.
- Task verification is one bounded retry loop (`max_verify_attempts`, no
  occlusion-aware "wait longer without burning an attempt" distinction) --
  see [FEATURE_COMPARISON.md](FEATURE_COMPARISON.md) for the full list of
  simplifications versus the reference demo's retry/deferral machinery.
- Staff have no live position tracking, so task-assignment "travel
  distance" is an in-zone/out-of-zone penalty (`app/roster.py`), not a
  geometric path length.
