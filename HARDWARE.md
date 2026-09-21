# Hardware Reference (SAMGRAHA / AUREX, SIH 2026 PS 26179)

This is a reference document, not code -- the software in this repo runs on
any laptop/webcam for demo purposes. Nothing here is simulated or partially
implemented in software (thermal sensors, UPS telemetry, PoE budgets, rack
servers); it's transcribed from the SIH pitch deck's hardware architecture
so the physical deployment plan lives next to the software that implements
its logic.

## Camera-to-cloud topology

```
[Shelf IP cameras]  [Overhead fisheye cameras]  [Checkout plaza cameras]
        \                    |                         /
         \___________________|________________________/
                             |  Cat6A PoE (802.3af/at), <=100m
                             v
              Managed L2+ Gigabit PoE+ switch
                 (isolated Vision VLAN)
                             |  10GbE SFP+ DAC
                             v
              Edge AI compute node
              - hardware video decode (NVDEC / V4L2)
              - TensorRT / ONNX inference
              - local SQLite + Mosquitto MQTT broker
                     |                    |
       (store LAN: metadata only)   (WAN: mTLS JSON, <50 Kbps)
                     v                    v
       Zebra handhelds, POS displays,   SD-WAN gateway -> corporate
       manager smartwatch, PA chime      cloud (fleet mgmt, dashboards)
```

Camera video never crosses onto the store's POS/payment or guest Wi-Fi
networks -- it's physically isolated on its own VLAN.

## Compute tiers

| Tier | Store size | Camera streams | Edge compute | AI throughput |
|---|---|---|---|---|
| MVP / Tier 1 | Convenience (1.5k-3k sq ft) | 4-8 | Intel N100 mini-PC or Jetson Orin Nano | 30+ FPS class-agnostic models |
| Tier 2 | Supermarket (15k-35k sq ft) | 16-32 | Jetson AGX Orin Industrial | ~275 TOPS/node |
| Tier 3 | Hypermarket (60k+ sq ft) | 48-128+ | 1-2U rack server, dual NVIDIA L4 | 600+ TOPS |

## Camera specs

- **Shelf-facing**: 2-4MP mini-dome/pinhole, 1080p @ 10-15 FPS, H.265, 120-140° wide FoV, PoE, true WDR. Mounted on the opposing aisle valence, 2.0-2.4m height, 35-45° depression angle. ~1 camera per 1.2-2.4m of shelf bay.
- **Overhead (queue/traffic)**: 4-6MP fisheye/wide dome, 1080p-2K @ 15 FPS, H.265, PoE, true WDR. Ceiling-mounted, 3.5-4.5m height, true nadir (90° down). ~1 camera per 2-3 checkout lanes, or per 10-15m of aisle.

## Calibration & auto-healing

Four ArUco/AprilTag markers, fixed at known cm positions on a shelf upright
or the floor, let the homography be **re-derived every evaluation tick**
instead of assumed static -- if the camera is bumped, the next tick that
sees all four markers self-corrects. Implemented in
[`app/vision/homography.py`](app/vision/homography.py) (`ArucoHomographyCalibrator`);
enable it per-camera in `store_layout.json`:

```json
"calibration": {
  "mode": "aruco",
  "markers": { "0": [0, 0], "1": [180, 0], "2": [180, 45], "3": [0, 45] }
}
```

## Network & power

- Managed L2+ Gigabit PoE+ switch, 30W/port budget (370-740W total switch budget).
- Cat6A shielded F/UTP cabling (refrigeration compressor EMI mitigation).
- 1000-1500VA online rackmount UPS with dry-contact/USB telemetry, wired to trigger a graceful shutdown (flush WAL, close SQLite cleanly) before power loss becomes data loss. This repo's SQLite layer already runs in **WAL mode** (`app/database.py`) so a hard power-cut mid-write can't corrupt the database -- the UPS integration itself is the only missing hardware-dependent half.

## Associate-facing hardware

- Rugged handheld scanners (e.g. Zebra TC5x) receive P1/P2 restock and
  planogram-drift tasks with bay/slot coordinates.
- Manager smartwatch / PA audio chime for P1 checkout-surge alerts.
- POS register screen pop-up for P2 predictive queue warnings.

None of the above associate-hardware integrations are implemented here --
this MVP's alert channels are Telegram + a generic webhook + the dashboard
WebSocket (`app/alerts/dispatcher.py`), which cover the same priority tiers
without needing the physical devices to demo the logic.
