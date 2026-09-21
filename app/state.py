"""
Thread-safe in-memory snapshot of the store's current state.

The vision pipeline (running on background threads, one per camera) writes
into this object; the FastAPI server (running on the asyncio event loop)
reads from it to answer REST requests and to broadcast periodic snapshots
over the /ws/live WebSocket. Plain-dict swaps under a lock are sufficient
here since updates are small and infrequent relative to the read cadence.
"""
import threading


class LiveState:
    def __init__(self):
        self._lock = threading.Lock()
        self.footfall = {"total_in": 0, "total_out": 0, "inside_count": 0}
        self.queues = {}       # lane_id -> queue status dict
        self.shelf_slots = {}  # slot_id -> shelf status dict

    def update_footfall(self, total_in: int, total_out: int):
        with self._lock:
            self.footfall = {
                "total_in": total_in,
                "total_out": total_out,
                "inside_count": max(0, total_in - total_out),
            }

    def update_queue(self, lane_id: str, state_dict: dict):
        with self._lock:
            self.queues[lane_id] = state_dict

    def update_shelf_slot(self, slot_id: str, state_dict: dict):
        with self._lock:
            self.shelf_slots[slot_id] = state_dict

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "footfall": dict(self.footfall),
                "queues": {k: dict(v) for k, v in self.queues.items()},
                "shelf_slots": {k: dict(v) for k, v in self.shelf_slots.items()},
            }


LIVE_STATE = LiveState()
