# Feature Comparison: this repo vs. "The Physical Ledger" reference demo

The reference is a single-file HTML/JS simulation (client-side only -- every
camera, POS terminal, and staff member is a JS object, not a real backend)
that models a much fuller product than the original MVP here implemented.
This document maps what it does to what this repo now does, split into
three buckets: **built** (real, tested Python logic), **partial** (the
concept exists but simplified), and **not implemented** (documented gap,
usually because it needs hardware or a real integration this repo has no
access to).

## Built

| Subsystem | Reference demo | This repo |
|---|---|---|
| Footfall tripwire | Entry/exit line-crossing counter | `app/vision/footfall.py` -- identical vector-crossing math |
| Queue dwell alerting | `L`, dwell threshold, confirmation window | `app/vision/queue_monitor.py` -- exact state machine |
| Congestion forecast | `CI = λ/(c·μ)`, open/close-lane recommendations | Same formula, same recommendation tiers, in `queue_monitor.py` |
| Shelf void detection | Classical edge-density backend (`sem_trained` unavailable, same as here) | `app/vision/shelf_monitor.py` -- same classical-CV approach, explicitly not claiming to be the trained YOLOv8-Seg model either |
| Planar homography (pixel -> cm) | `x' = Hx` | `app/vision/homography.py`, `PlanarHomography` |
| Camera-drift auto-healing | ORB + RANSAC reprojection error, recalibration | `ArucoHomographyCalibrator` -- simpler (ArUco markers, not raw ORB feature matching), same self-healing intent |
| Movement/dwell heatmaps | 2D Gaussian accumulator, traffic vs. dwell layers | `app/vision/heatmap.py` -- same two-layer decaying accumulator |
| **Fusion / cause classification** | Shelf state x backroom x system inventory x POS velocity -> `REPLENISHMENT_REQUIRED` / `PURCHASING_ALERT` / `LOW_PRIORITY_OOS` / `VISUAL_ALERT_ONLY` | `app/fusion.py` -- same truth table, same order (LOW velocity checked before backroom) |
| **Bounded priority score** | `raw = (V/V_ref)*M*(G/G_ref)*A/(1+D)`, clamped 0-100, bucketed P0-P3 | `app/fusion.py::compute_priority` -- same formula, same bucket thresholds |
| **Task lifecycle** | OPEN -> ASSIGNED -> STARTED -> VERIFYING -> RESOLVED / VERIFY_UNAVAILABLE, SLA deadlines | `app/tasks.py` -- simplified retry/deferral machinery (single bounded retry loop instead of 6-attempt/3-cycle nesting), same core guarantee: a tap never resolves a task, only a re-check of the live signal does |
| **Staff roster & assignment** | Zone match x busy-penalty x workload scoring | `app/roster.py` -- same scoring shape, travel distance approximated as an in-zone/out-of-zone penalty instead of a geometric path length (no staff position tracking in this repo) |
| **Mock POS adapter** | Simulated transaction log, velocity vs. 14-day baseline | `app/pos.py::MockPOSAdapter` -- rolling-window velocity vs. a configured baseline, same HIGH/NORMAL/LOW multipliers (1.5x / 0.5x); real integration point is `POSAdapter` (ABC) |
| **Mock inventory adapter** | Backroom + system-of-record counts | `app/inventory.py::MockInventoryAdapter` -- same two-count model; real integration point is `InventoryAdapter` (ABC) |
| Zero-PII perception | No facial recognition, ephemeral track IDs | Same design throughout (`app/vision/tracker.py`) |
| WAL-mode SQLite | "Local store: SQLite, WAL journal" | `app/database.py` -- actually running in WAL mode, not simulated |

## Partial

| Subsystem | Reference demo | This repo | Gap |
|---|---|---|---|
| Human occlusion gating | Slot-level *and* frame-level (`occl_frame` -> `FRAME_REJECT`), `valid_fraction_min` gates an `UNKNOWN` state | `shelf_monitor.py` has slot-level occlusion only (cached last-state reuse) | No frame-wide reject state, no explicit `UNKNOWN` (low-confidence) state distinct from "void" |
| Density / heatmap | Grid cells owned by exactly one camera, hysteresis states (enter/exit thresholds + duration), 14-day baseline comparison, persisted windowed aggregates (`window_uuid`, `person_seconds`, `distinct_tracks`, ...) | `heatmap.py` is a plain decaying accumulator, no cell ownership, no hysteresis, no baseline, nothing persisted | Would need a proper floor-plan grid + per-cell camera-ownership resolution, which needs a calibrated multi-camera floor homography this repo doesn't build |
| Verification retry policy | 6 attempts/cycle, 3 cycles, distinct `VERIFY_DEFERRED` (blocked view) vs. `UNRESOLVED` (clear view, still broken) states | `tasks.py` has one bounded retry loop (`max_verify_attempts`, default 6) -> `VERIFY_UNAVAILABLE` | No occlusion-aware "wait longer, don't burn an attempt" distinction |
| POS clock normalization | Offset estimation from REST `Date` header, `POS_CLOCK_SUSPECT` at >5s drift, backfill on reconnect | `MockPOSAdapter` has no clock model at all -- everything is node-local time | Not needed until there's a real POS with its own clock |
| Shelf-tag OCR | Read + matched against the planogram slot map, flags `LABEL_NOT_FOUND` / `UNEXPECTED_SKU` / `INCORRECT_SHELF_ROW` | `app/vision/ocr.py` reads text (optional, `pytesseract`) but never compares it to an expected slot map | No planogram engine to compare against (see below) |

## Not implemented (documented, not hidden)

| Subsystem | Why it's out of scope here |
|---|---|
| Planogram engine (expected vs. observed slot map, drift findings) | Needs a maintained digital planogram feed from a real merchandising system; nothing to compare OCR reads against without one |
| Full queue arrival-lag cross-correlation fitting | The demo fits entrance-to-queue-join lag from paired historical counts over hours of data; this repo uses a fixed `arrival_window_seconds` instead -- documented simplification in `queue_monitor.py` |
| Store-and-forward outbox with idempotent UUID sync | No central cloud endpoint exists to sync to; `AlertDispatcher`'s webhook/Telegram channels are best-effort and don't queue-and-retry |
| Reports tab (daily/weekly aggregates, CSV export) | Straightforward to add on top of the existing SQLite tables, just not built yet |
| Acceptance test harness (22 scripted scenarios + 4 "H" tests) | This repo's `tests/test_pipeline_smoke.py` (20 tests) covers the equivalent *logic* per subsystem; it isn't organized as a scenario-replay harness against a full store simulator |
| Model calibration pipeline (train/val/test threshold selection, frozen `tau`) | `config.py`'s `void_ratio_alert` is a fixed constant; no calibration dataset or split exists to derive it from |
| RKNN/QNN deployment verification, hardware BOM | Physical hardware -- see [HARDWARE.md](HARDWARE.md) |
| Thermal/UPS monitoring, graceful shutdown on power loss | Physical hardware; the one piece that *is* implemented (WAL-mode SQLite) is the software half of "survive a hard power cut" |
| Staff live position tracking | No BLE/handheld locating integration; `app/roster.py` substitutes a zone-membership penalty for travel distance |

## What to look at first

- `app/fusion.py` + `app/tasks.py` -- the actual "mock backend to show basic
  working" this comparison was written to explain. Wired into
  `main.py::run_camera_pipeline` (shelf out-of-stock -> fusion -> task) and
  the queue congestion block (critical forecast -> lane-open task).
- `app/pos.py` / `app/inventory.py` -- both are one-file mocks behind an
  `ABC` (`POSAdapter`, `InventoryAdapter`); a real integration replaces the
  mock class, not the callers.
- `tests/test_pipeline_smoke.py` -- run it (`python tests/test_pipeline_smoke.py`)
  for a fast, camera-free check that the truth tables, priority formula,
  and task state machine all behave as documented above.
