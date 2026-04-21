"""
utils/cpu_monitor.py
====================
Adaptive CPU-usage guard for the background webcam pipeline.

Goal: keep total process CPU usage ≤ TARGET_CPU_PCT (default 10%).

Strategy
--------
1. A `CPUGuard` background thread samples `psutil.Process.cpu_percent()`
   every SAMPLE_INTERVAL seconds.
2. If usage exceeds the target it increments a shared `skip_frames`
   counter (max MAX_SKIP).  EyeTracker consults this counter and drops
   frames accordingly.
3. If usage falls below the target the counter is decremented (min 0).
4. The guard also monitors total system CPU and backs off if the host
   is under heavy load from other processes.

The frame-skip counter is exposed as a thread-safe `threading.Event`-
based token so callers don't need locks.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed — CPU guard disabled. "
                   "Install with: pip install psutil")


TARGET_CPU_PCT   = 10.0    # % of one core
SAMPLE_INTERVAL  = 2.0     # seconds between measurements
MAX_SKIP         = 5       # maximum frames to skip per capture cycle
SYSTEM_CPU_LIMIT = 80.0    # back off harder if system is >80% busy


class FrameSkipController:
    """
    Shared state object read by EyeTracker, written by CPUGuard.

    Thread-safe via an atomic integer (GIL-protected in CPython, and
    we only do single-value reads/writes which are atomic on CPython).
    """

    def __init__(self) -> None:
        self._skip_count: int = 0     # 0 = no skip, N = skip N frames

    @property
    def skip_count(self) -> int:
        return self._skip_count

    @skip_count.setter
    def skip_count(self, v: int) -> None:
        self._skip_count = max(0, min(v, MAX_SKIP))

    def should_skip(self, frame_index: int) -> bool:
        """
        Returns True if `frame_index` should be dropped.
        With skip_count=2, every 3rd frame is processed (indices 0,3,6…).
        """
        n = self._skip_count
        if n == 0:
            return False
        return (frame_index % (n + 1)) != 0


class CPUGuard:
    """
    Background thread that monitors process CPU and adjusts
    `FrameSkipController.skip_count` automatically.

    Usage
    -----
    >>> controller = FrameSkipController()
    >>> guard = CPUGuard(controller, target_pct=10.0)
    >>> guard.start()
    # …
    >>> guard.stop()
    """

    def __init__(
        self,
        controller: FrameSkipController,
        target_pct: float = TARGET_CPU_PCT,
    ) -> None:
        self._ctrl    = controller
        self._target  = target_pct
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc    = psutil.Process(os.getpid()) if _PSUTIL_AVAILABLE else None

        # Exponential moving average of CPU usage
        self._ema_cpu : float = 0.0
        self._ema_alpha: float = 0.4   # smoothing factor

    def start(self) -> None:
        if not _PSUTIL_AVAILABLE:
            logger.warning("CPUGuard cannot start — psutil unavailable.")
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="CPUGuardThread"
        )
        self._thread.start()
        logger.info("CPUGuard started (target=%.1f%%).", self._target)

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3.0)

    @property
    def ema_cpu(self) -> float:
        return self._ema_cpu

    # ------------------------------------------------------------------
    def _monitor_loop(self) -> None:
        # Prime the psutil measurement (first call always returns 0.0)
        if self._proc:
            self._proc.cpu_percent(interval=None)
        time.sleep(SAMPLE_INTERVAL)

        while self._running.is_set():
            proc_cpu   = self._proc.cpu_percent(interval=None) if self._proc else 0.0
            system_cpu = psutil.cpu_percent(interval=None)

            # EMA smoothing
            self._ema_cpu = (self._ema_alpha * proc_cpu
                             + (1 - self._ema_alpha) * self._ema_cpu)

            old_skip = self._ctrl.skip_count

            if self._ema_cpu > self._target * 1.5 or system_cpu > SYSTEM_CPU_LIMIT:
                # Aggressively increase skip
                self._ctrl.skip_count = old_skip + 2
            elif self._ema_cpu > self._target:
                # Gently increase skip
                self._ctrl.skip_count = old_skip + 1
            elif self._ema_cpu < self._target * 0.6:
                # Recover — process fewer skips
                self._ctrl.skip_count = old_skip - 1

            new_skip = self._ctrl.skip_count
            if new_skip != old_skip:
                logger.debug(
                    "CPU=%.1f%% (EMA=%.1f%%)  skip_count: %d → %d",
                    proc_cpu, self._ema_cpu, old_skip, new_skip,
                )

            time.sleep(SAMPLE_INTERVAL)


# ---------------------------------------------------------------------------
# Integration helper for EyeTracker
# ---------------------------------------------------------------------------
def patch_eye_tracker_with_cpu_guard(
    eye_tracker,
    controller: FrameSkipController,
) -> None:
    """
    Monkey-patches EyeTracker._capture_loop to use the FrameSkipController.

    This is a non-invasive way to add frame-skipping without modifying
    EyeTracker's source — useful for testing or if you prefer a clean
    separation of concerns.

    In production you can instead pass `controller` directly to
    EyeTracker and have it call `controller.should_skip(frame_index)`.
    """
    original_loop = eye_tracker._capture_loop.__func__

    def patched_loop(self_et) -> None:  # type: ignore[override]
        import cv2, time as _time, queue as _q
        import mediapipe as mp

        self_et._mp_face_mesh = mp.solutions.face_mesh
        self_et._face_mesh = self_et._mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(self_et.camera_index)
        if not cap.isOpened():
            self_et._running.clear()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self_et.target_fps)

        frame_interval = 1.0 / self_et.target_fps
        frame_index    = 0

        try:
            while self_et._running.is_set():
                loop_start = _time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    _time.sleep(0.01)
                    continue

                frame_index += 1

                if controller.should_skip(frame_index):
                    # Still need to consume the frame for the cap buffer
                    pass
                else:
                    eye_frame = self_et._process_frame(frame)
                    try:
                        self_et.frame_queue.put_nowait(eye_frame)
                    except _q.Full:
                        pass

                if self_et.show_preview:
                    self_et._draw_overlay(frame, eye_frame if not controller.should_skip(frame_index) else self_et._process_frame(frame))
                    cv2.imshow("CognitiveMonitor – Eye Tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                elapsed   = _time.monotonic() - loop_start
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    _time.sleep(sleep_for)
        finally:
            cap.release()
            self_et._face_mesh.close()

    import types
    eye_tracker._capture_loop = types.MethodType(patched_loop, eye_tracker)
    logger.debug("EyeTracker patched with FrameSkipController.")
