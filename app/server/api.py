"""
Local Store UI & Enterprise Integration surface (Section 4 MVP table:
"Lightweight Web Dashboard (FastAPI + HTML5)").

Exposes:
  - REST endpoints for current footfall / queue / shelf state and recent
    history, read from the shared LiveState + the SQLite database.
  - A WebSocket (/ws/live) that pushes periodic state snapshots and
    real-time alert events to the dashboard, avoiding client-side polling.
  - The single-page dashboard itself at "/".

This module only reads from LIVE_STATE and the database; all writes happen
on the vision pipeline's background threads (see main.py).
"""
import asyncio
import queue
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response

from app.database import Database
from app.alerts.dispatcher import AlertDispatcher
from app.state import LIVE_STATE
from config import CONFIG

logger = logging.getLogger("api")
STATIC_DIR = Path(__file__).parent / "static"


def create_app(db: Database, dispatcher: AlertDispatcher) -> FastAPI:
    app = FastAPI(title="Edge Retail Intelligence Platform", version="0.1.0")

    alert_queue: "queue.Queue[dict]" = queue.Queue()
    dispatcher.register_listener(lambda event: alert_queue.put(event.to_dict()))

    connections: List[WebSocket] = []

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return (STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "store_id": CONFIG.store_id, "gateway_id": CONFIG.gateway_id}

    @app.get("/api/footfall/summary")
    async def footfall_summary():
        return LIVE_STATE.snapshot()["footfall"]

    @app.get("/api/footfall/recent")
    async def footfall_recent(limit: int = 50):
        return db.recent_footfall(limit=limit)

    @app.get("/api/queue/status")
    async def queue_status():
        return LIVE_STATE.snapshot()["queues"]

    @app.get("/api/shelf/status")
    async def shelf_status():
        return LIVE_STATE.snapshot()["shelf_slots"]

    @app.get("/api/shelf/alerts")
    async def shelf_alerts(limit: int = 100):
        return db.active_shelf_alerts(limit=limit)

    @app.get("/api/heatmap/{camera_name}/{layer}.png")
    async def heatmap_png(camera_name: str, layer: str):
        if layer not in ("traffic", "dwell"):
            raise HTTPException(status_code=400, detail="layer must be 'traffic' or 'dwell'")
        png_bytes = LIVE_STATE.get_heatmap_png(camera_name, layer)
        if png_bytes is None:
            raise HTTPException(status_code=404, detail=f"no heatmap yet for camera '{camera_name}'")
        return Response(content=png_bytes, media_type="image/png")

    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket):
        await websocket.accept()
        connections.append(websocket)
        try:
            await websocket.send_json({"type": "state", "data": LIVE_STATE.snapshot()})
            while True:
                # The dashboard never sends anything; this simply blocks
                # until the client disconnects so we can clean it up.
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            if websocket in connections:
                connections.remove(websocket)

    async def _broadcast(message: dict):
        stale = []
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                stale.append(ws)
        for ws in stale:
            if ws in connections:
                connections.remove(ws)

    async def _state_broadcaster():
        while True:
            await asyncio.sleep(2.0)
            await _broadcast({"type": "state", "data": LIVE_STATE.snapshot()})

    async def _alert_broadcaster():
        while True:
            event_dict = await asyncio.to_thread(alert_queue.get)
            await _broadcast({"type": "alert", "data": event_dict})

    @app.on_event("startup")
    async def _startup():
        asyncio.create_task(_state_broadcaster())
        asyncio.create_task(_alert_broadcaster())

    return app
