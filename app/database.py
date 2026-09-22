"""
Embedded SQLite persistence layer (Section 5.1 of the spec).

Schema is reproduced verbatim from the platform specification so that the
edge-to-cloud sync contract (Section 5.2) can be built on top of it without
translation. WAL mode is enabled for concurrent reader (dashboard) / writer
(vision pipeline) access, matching the "ACID compliant, resilient to power
loss" rationale in the spec's MVP hardware table.
"""
import json
import sqlite3
import threading
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS footfall_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    direction TEXT CHECK(direction IN ('IN', 'OUT')),
    dwell_duration_seconds REAL,
    synced INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS shelf_compliance_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    aisle_id TEXT NOT NULL,
    shelf_tier INTEGER NOT NULL,
    slot_id TEXT NOT NULL,
    sku_id TEXT NOT NULL,
    void_ratio REAL NOT NULL,
    alert_triggered INTEGER DEFAULT 0,
    synced INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS queue_telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    lane_id TEXT NOT NULL,
    people_count INTEGER NOT NULL,
    estimated_wait_seconds REAL,
    synced INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    reason TEXT,
    zone TEXT,
    cause TEXT,
    priority_score INTEGER,
    priority_bucket TEXT,
    slot_id TEXT,
    sku_id TEXT,
    lane_id TEXT,
    staff_id TEXT,
    assignee TEXT,
    status TEXT NOT NULL,
    resolution_state TEXT NOT NULL,
    sla_breached INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    assigned_at REAL,
    started_at REAL,
    completed_at REAL,
    resolved_at REAL,
    verify_at REAL,
    verify_attempts INTEGER DEFAULT 0,
    sla_deadline REAL,
    history_json TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

_lock = threading.Lock()


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def insert_footfall_event(self, direction: str, dwell_duration_seconds: float = None):
        with _lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO footfall_events (direction, dwell_duration_seconds) VALUES (?, ?)",
                (direction, dwell_duration_seconds),
            )

    def insert_shelf_event(self, aisle_id: str, shelf_tier: int, slot_id: str, sku_id: str,
                            void_ratio: float, alert_triggered: bool):
        with _lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO shelf_compliance_events
                   (aisle_id, shelf_tier, slot_id, sku_id, void_ratio, alert_triggered)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (aisle_id, shelf_tier, slot_id, sku_id, void_ratio, int(alert_triggered)),
            )

    def insert_queue_telemetry(self, lane_id: str, people_count: int, estimated_wait_seconds: float):
        with _lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO queue_telemetry (lane_id, people_count, estimated_wait_seconds)
                   VALUES (?, ?, ?)""",
                (lane_id, people_count, estimated_wait_seconds),
            )

    def recent_footfall(self, limit: int = 50):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM footfall_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def footfall_counts_since(self, since_iso: str):
        with self._connect() as conn:
            row = conn.execute(
                """SELECT
                       SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) AS inbound,
                       SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) AS outbound
                   FROM footfall_events WHERE timestamp >= ?""",
                (since_iso,),
            ).fetchone()
            return {"inbound": row["inbound"] or 0, "outbound": row["outbound"] or 0}

    def recent_queue_telemetry(self, lane_id: str = None, limit: int = 50):
        with self._connect() as conn:
            if lane_id:
                rows = conn.execute(
                    "SELECT * FROM queue_telemetry WHERE lane_id = ? ORDER BY id DESC LIMIT ?",
                    (lane_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM queue_telemetry ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def active_shelf_alerts(self, limit: int = 100):
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM shelf_compliance_events
                   WHERE alert_triggered = 1 ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def upsert_task(self, task: dict):
        with _lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO tasks (task_id, kind, title, reason, zone, cause, priority_score,
                       priority_bucket, slot_id, sku_id, lane_id, staff_id, assignee, status,
                       resolution_state, sla_breached, created_at, assigned_at, started_at,
                       completed_at, resolved_at, verify_at, verify_attempts, sla_deadline,
                       history_json, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(task_id) DO UPDATE SET
                       staff_id=excluded.staff_id, assignee=excluded.assignee, status=excluded.status,
                       resolution_state=excluded.resolution_state, sla_breached=excluded.sla_breached,
                       assigned_at=excluded.assigned_at, started_at=excluded.started_at,
                       completed_at=excluded.completed_at, resolved_at=excluded.resolved_at,
                       verify_at=excluded.verify_at, verify_attempts=excluded.verify_attempts,
                       history_json=excluded.history_json, updated_at=CURRENT_TIMESTAMP""",
                (
                    task["task_id"], task["kind"], task["title"], task["reason"], task["zone"],
                    task["cause"], task["priority_score"], task["priority_bucket"], task["slot_id"],
                    task["sku_id"], task["lane_id"], task["staff_id"], task["assignee"], task["status"],
                    task["resolution_state"], int(task["sla_breached"]), task["created_at"],
                    task["assigned_at"], task["started_at"], task["completed_at"], task["resolved_at"],
                    task["verify_at"], task["verify_attempts"], task["sla_deadline"],
                    json.dumps(task["history"]),
                ),
            )

    def list_tasks(self, open_only: bool = False, limit: int = 200):
        with self._connect() as conn:
            if open_only:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE status != 'CLOSED' ORDER BY priority_score DESC, created_at ASC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["history"] = json.loads(d.pop("history_json") or "[]")
                results.append(d)
            return results

    def unsynced_batch(self, table: str, limit: int = 200):
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM {table} WHERE synced = 0 ORDER BY id ASC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_synced(self, table: str, ids):
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with _lock, self._connect() as conn:
            conn.execute(f"UPDATE {table} SET synced = 1 WHERE id IN ({placeholders})", ids)
