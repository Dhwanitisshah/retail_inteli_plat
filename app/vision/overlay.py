"""
Optional debug overlay (--show) for local demoing: draws tripwires, queue
ROIs, shelf slots, and live track markers directly on the frame. Never used
in headless/production operation and never persisted to disk, consistent
with the platform's "no raw video retention" privacy design -- this window
is strictly a live, local, in-memory preview.
"""
import cv2

COLOR_TRIPWIRE = (255, 200, 0)
COLOR_QUEUE = (0, 200, 255)
COLOR_SHELF_OK = (0, 200, 0)
COLOR_SHELF_ALERT = (0, 0, 255)
COLOR_TRACK = (255, 255, 255)


def draw_tripwires(frame, tripwires):
    for tw in tripwires:
        cv2.line(frame, tuple(map(int, tw.p1)), tuple(map(int, tw.p2)), COLOR_TRIPWIRE, 2)
        cv2.putText(frame, tw.name, tuple(map(int, tw.p1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOR_TRIPWIRE, 1, cv2.LINE_AA)


def draw_polygons(frame, polygons):
    for poly in polygons:
        pts = [tuple(map(int, p)) for p in poly.points]
        for i in range(len(pts)):
            cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], COLOR_QUEUE, 2)
        cv2.putText(frame, poly.name, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOR_QUEUE, 1, cv2.LINE_AA)


def draw_shelf_slots(frame, slot_states):
    for state in slot_states:
        x, y, w, h = state.slot.rect
        color = COLOR_SHELF_ALERT if state.out_of_stock_alert else (
            (0, 165, 255) if state.low_stock_warning else COLOR_SHELF_OK
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{state.slot.slot_id} void={state.void_ratio:.2f}"
        if state.gap_cm is not None:
            label += f" gap {state.gap_cm:.0f}cm"
        cv2.putText(frame, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)


def draw_tracks(frame, tracks):
    for t in tracks:
        x, y, w, h = t.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_TRACK, 1)
        cv2.putText(frame, f"#{t.track_id}", (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    COLOR_TRACK, 1, cv2.LINE_AA)
