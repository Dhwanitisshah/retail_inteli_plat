"""
Task & verification engine, simplified from the reference demo's task state
machine. A fused alert (or a critical queue-congestion forecast) becomes a
Task assigned to a rostered staff member; once marked done, the *same*
camera signal that raised the alert re-checks the slot/lane on a timer
rather than trusting a tap on a handheld -- closing the loop the base MVP
never had (it could alert, but never knew whether anyone fixed anything).

Two verification paths, both simplified from the reference demo's full
retry/deferral-cycle machinery into a single bounded retry loop:
  - After being marked done: re-check after `verify_after_seconds`; if not
    resolved, retry every `recheck_interval_seconds` up to `max_attempts`,
    then give up as VERIFY_UNAVAILABLE (escalate to a human, not the same
    as "still broken").
  - Independently, `sla_breached` flips true if a task is still open past
    its SLA deadline, whatever state it's in -- a task can still resolve
    afterwards (late), it just carries that flag forever.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from app.database import Database
from app.fusion import PriorityResult
from app.roster import Roster

STATUS_OPEN = "OPEN"
STATUS_ASSIGNED = "ASSIGNED"
STATUS_STARTED = "STARTED"
STATUS_VERIFYING = "VERIFYING"
STATUS_CLOSED = "CLOSED"

RESOLUTION_OPEN = "OPEN"
RESOLUTION_RESOLVED = "RESOLVED"
RESOLUTION_VERIFY_UNAVAILABLE = "VERIFY_UNAVAILABLE"

DEFAULT_SLA_MINUTES = {"restock": 10, "audit": 30, "check": 15, "lane": 5}


@dataclass
class Task:
    task_id: str
    kind: str
    title: str
    reason: str
    zone: str
    cause: str
    priority: PriorityResult
    slot_id: Optional[str] = None
    sku_id: Optional[str] = None
    lane_id: Optional[str] = None
    staff_id: Optional[str] = None
    assignee: Optional[str] = None
    status: str = STATUS_OPEN
    resolution_state: str = RESOLUTION_OPEN
    sla_breached: bool = False
    created_at: float = field(default_factory=time.time)
    assigned_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    resolved_at: Optional[float] = None
    verify_at: Optional[float] = None
    verify_attempts: int = 0
    sla_deadline: float = 0.0
    history: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "kind": self.kind, "title": self.title, "reason": self.reason,
            "zone": self.zone, "cause": self.cause, "priority_score": self.priority.score,
            "priority_bucket": self.priority.bucket, "slot_id": self.slot_id, "sku_id": self.sku_id,
            "lane_id": self.lane_id, "staff_id": self.staff_id, "assignee": self.assignee,
            "status": self.status, "resolution_state": self.resolution_state,
            "sla_breached": self.sla_breached, "created_at": self.created_at,
            "assigned_at": self.assigned_at, "started_at": self.started_at,
            "completed_at": self.completed_at, "resolved_at": self.resolved_at,
            "verify_at": self.verify_at, "verify_attempts": self.verify_attempts,
            "sla_deadline": self.sla_deadline, "history": self.history,
        }


class TaskManager:
    def __init__(self, roster: Roster, db: Optional[Database] = None,
                 sla_minutes: Dict[str, int] = None, verify_after_seconds: float = 60.0,
                 recheck_interval_seconds: float = 30.0, max_verify_attempts: int = 6,
                 on_event: Optional[Callable[[str, Task], None]] = None):
        self.roster = roster
        self.db = db
        self.sla_minutes = sla_minutes or DEFAULT_SLA_MINUTES
        self.verify_after_seconds = verify_after_seconds
        self.recheck_interval_seconds = recheck_interval_seconds
        self.max_verify_attempts = max_verify_attempts
        self.on_event = on_event
        self._tasks: Dict[str, Task] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def _log(self, task: Task, status: str, note: str):
        task.history.append({"t": time.time(), "status": status, "note": note})
        if self.on_event:
            self.on_event(status, task)
        if self.db:
            self.db.upsert_task(task.to_dict())

    def create_task(self, kind: str, zone: str, title: str, reason: str, cause: str,
                     priority: PriorityResult, slot_id: str = None, sku_id: str = None,
                     lane_id: str = None) -> Task:
        with self._lock:
            self._seq += 1
            task = Task(
                task_id=f"T-{self._seq:04d}", kind=kind, title=title, reason=reason, zone=zone,
                cause=cause, priority=priority, slot_id=slot_id, sku_id=sku_id, lane_id=lane_id,
                sla_deadline=time.time() + self.sla_minutes.get(kind, 15) * 60.0,
            )
            self._tasks[task.task_id] = task
            self._log(task, STATUS_OPEN, f"Created ({cause}, {priority.bucket} {priority.score})")

            staff = self.roster.best_candidate(zone)
            if staff:
                self.roster.enqueue(staff.staff_id, task.task_id)
                task.staff_id, task.assignee = staff.staff_id, staff.name
                task.status = STATUS_ASSIGNED
                task.assigned_at = time.time()
                self._log(task, STATUS_ASSIGNED, f"Assigned to {staff.name}")
            return task

    def mark_started(self, task_id: str) -> Optional[Task]:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status == STATUS_CLOSED:
                return None
            task.status = STATUS_STARTED
            task.started_at = time.time()
            self._log(task, STATUS_STARTED, f"{task.assignee or 'Staff'} started")
            return task

    def mark_completed(self, task_id: str) -> Optional[Task]:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status == STATUS_CLOSED:
                return None
            if task.status != STATUS_STARTED:
                task.status = STATUS_STARTED
                task.started_at = task.started_at or time.time()
                self._log(task, STATUS_STARTED, f"{task.assignee or 'Staff'} started")
            task.status = STATUS_VERIFYING
            task.completed_at = time.time()
            task.verify_at = task.completed_at + self.verify_after_seconds
            self._log(task, STATUS_VERIFYING, "Marked done; camera will re-check")
            return task

    def _close(self, task: Task, resolution_state: str, note: str):
        task.resolution_state = resolution_state
        task.status = STATUS_CLOSED
        task.verify_at = None
        if resolution_state == RESOLUTION_RESOLVED:
            task.resolved_at = time.time()
        if task.staff_id:
            self.roster.release(task.staff_id, task.task_id)
        self._log(task, resolution_state, note)

    def tick(self, is_resolved: Callable[[Task], bool]):
        """Call periodically. Runs due verifications and SLA-deadline checks
        across every open task. `is_resolved(task)` re-checks the live
        camera/queue signal that originally raised the alert."""
        now = time.time()
        with self._lock:
            tasks_snapshot = list(self._tasks.values())
        for task in tasks_snapshot:
            if task.status == STATUS_CLOSED:
                continue

            if task.status == STATUS_VERIFYING and task.verify_at is not None and now >= task.verify_at:
                if is_resolved(task):
                    self._close(task, RESOLUTION_RESOLVED,
                                f"Confirmed by camera{' (late)' if task.sla_breached else ''}")
                else:
                    task.verify_attempts += 1
                    if task.verify_attempts >= self.max_verify_attempts:
                        self._close(task, RESOLUTION_VERIFY_UNAVAILABLE,
                                     f"Not confirmed after {task.verify_attempts} attempts -- needs a manual check")
                    else:
                        task.verify_at = now + self.recheck_interval_seconds
                        self._log(task, STATUS_VERIFYING,
                                   f"Still not resolved; retry {task.verify_attempts}/{self.max_verify_attempts}")

            if not task.sla_breached and now >= task.sla_deadline and task.status != STATUS_CLOSED:
                task.sla_breached = True
                self._log(task, task.status, f"SLA deadline passed ({self.sla_minutes.get(task.kind, 15)} min)")

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_tasks(self, open_only: bool = False) -> List[Task]:
        tasks = list(self._tasks.values())
        if open_only:
            tasks = [t for t in tasks if t.status != STATUS_CLOSED]
        return sorted(tasks, key=lambda t: (-t.priority.score, t.created_at))
