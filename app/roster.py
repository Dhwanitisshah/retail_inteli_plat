"""
Staff roster & task assignment scoring, simplified from the reference demo's
`assignScore()`. Real position/travel-distance tracking would need staff to
carry a locating device (handheld, BLE badge, etc.) that this repo has no
integration for, so travel distance collapses to a simple in-zone/out-of-zone
penalty rather than a geometric path length -- documented simplification,
not a hidden one.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Staff:
    staff_id: str
    name: str
    zones: List[str]  # aisle_id / "CHECKOUT" values this person covers
    active_task_id: Optional[str] = None
    queued_task_ids: List[str] = field(default_factory=list)

    @property
    def workload(self) -> int:
        return (1 if self.active_task_id else 0) + len(self.queued_task_ids)


class Roster:
    def __init__(self, staff: List[Staff]):
        self.staff = {s.staff_id: s for s in staff}

    def best_candidate(self, zone: str) -> Optional[Staff]:
        candidates = list(self.staff.values())
        if not candidates:
            return None

        def score(s: Staff) -> float:
            in_zone = zone in s.zones
            busy = s.active_task_id is not None
            return (1.0 if in_zone else 0.5) * (0.35 if busy else 1.0) / (1 + s.workload)

        return max(candidates, key=lambda s: (score(s), s.name), default=None)

    def enqueue(self, staff_id: str, task_id: str) -> None:
        s = self.staff[staff_id]
        if s.active_task_id is None:
            s.active_task_id = task_id
        else:
            s.queued_task_ids.append(task_id)

    def release(self, staff_id: str, task_id: str) -> None:
        s = self.staff.get(staff_id)
        if not s:
            return
        if s.active_task_id == task_id:
            s.active_task_id = s.queued_task_ids.pop(0) if s.queued_task_ids else None
        elif task_id in s.queued_task_ids:
            s.queued_task_ids.remove(task_id)
