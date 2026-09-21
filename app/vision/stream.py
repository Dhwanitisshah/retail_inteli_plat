"""
Frame ingestion (Section 2 / Module: Frame Ingestion & Conditioning).

Wraps cv2.VideoCapture so the rest of the pipeline is agnostic to whether the
source is a webcam index, a local video file (used for demos/testing), or an
RTSP/ONVIF camera URL. No frames are ever written to disk here -- each frame
is handed to the caller and then goes out of scope, matching the "immediate
memory-frame release" requirement in the spec.
"""
import time
import cv2


class FrameSource:
    def __init__(self, source, loop: bool = False, target_fps: float = None):
        """
        source: int (webcam index), str path to a video file, or an RTSP/HTTP URL.
        loop: if True and source is a file, restart from frame 0 at EOF (handy for demos).
        target_fps: if set, sleeps to approximate this capture rate instead of
                    draining a file as fast as the CPU can decode it.
        """
        self.source = source
        self.loop = loop
        self.target_fps = target_fps
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source!r}")
        self._is_file = isinstance(source, str) and not source.lower().startswith(
            ("rtsp://", "http://", "https://")
        )

    def frames(self):
        frame_interval = 1.0 / self.target_fps if self.target_fps else None
        while True:
            t_start = time.time()
            ok, frame = self._cap.read()
            if not ok:
                if self.loop and self._is_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            yield frame
            if frame_interval:
                elapsed = time.time() - t_start
                remaining = frame_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)

    def release(self):
        self._cap.release()
