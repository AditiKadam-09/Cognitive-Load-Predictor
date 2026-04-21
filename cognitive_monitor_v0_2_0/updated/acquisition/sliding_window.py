"""
acquisition/sliding_window.py
==============================
Sliding-window aggregator for eye-tracking, face-expression, and
keystroke events.

Three-signal fusion layout
---------------------------
The `WindowedFeatures.feature_vector` property now returns a 17-element
array (previously 14).  Three new face-signal features sit between the
existing eye and keystroke blocks:

  Indices 0–6   → Eye signal    (EAR, blink, gaze, PERCLOS, …)
  Indices 7–9   → Face signal   (brow furrow, mouth open, eye asymmetry)  NEW
  Indices 10–16 → Keystroke signal

Design
------
A single SlidingWindow instance accepts raw events from both sensors.
Every `step_size` seconds it fires a callback with a WindowedFeatures
object containing statistical summaries for the past `window_size` seconds.

                      window_size (e.g. 7 s)
             ┌────────────────────────────────┐
  ────────────────────────────────────────────────────→ time
                  ↑                           ↑
             window_start                window_end
                           ← step_size →
                                ↑
                           next trigger
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional
from collections import deque

import numpy as np

from .eye_tracker      import EyeFrame, EyeWindowStats
from .keystroke_monitor import (
    KeystrokeEvent, KeystrokeWindowStats, KeyCategory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Combined output dataclass
# ---------------------------------------------------------------------------
@dataclass
class WindowedFeatures:
    """
    All features from one sliding-window interval, ready for the ML fusion model.

    The `feature_vector` property returns a flat numpy float32 array of shape (17,).
    """
    timestamp:  float
    eye_stats:  EyeWindowStats
    key_stats:  KeystrokeWindowStats

    @property
    def feature_vector(self) -> np.ndarray:
        """
        Returns a 1-D numpy float32 array of shape (17,).

        Feature order (document and preserve this):
          ── Eye signal ──────────────────────────────────────
          0   mean_ear
          1   std_ear
          2   blink_rate           (blinks/min)
          3   gaze_stability       (0–1, higher = more stable)
          4   perclos              (0–1, fraction closed)
          5   head_roll_mean       (degrees)
          6   face_coverage        (0–1)
          ── Face signal (NEW) ────────────────────────────────
          7   brow_furrow_ratio    (0–1; lower = more furrowed = more stress)
          8   mouth_open_ratio     (MAR; extreme high OR low = stress)
          9   eye_asymmetry        (|left_ear - right_ear|; higher = more stress)
          ── Keystroke signal ─────────────────────────────────
          10  typing_speed_cps
          11  mean_iki             (seconds)
          12  std_iki              (seconds)
          13  cv_iki               (dimensionless)
          14  backspace_rate       (0–1)
          15  burst_coefficient    (0–1)
          16  pause_count          (integer cast to float)
        """
        e = self.eye_stats
        k = self.key_stats
        return np.array([
            # Eye signal
            e.mean_ear,
            e.std_ear,
            e.blink_rate,
            e.gaze_stability,
            e.perclos,
            e.head_roll_mean,
            e.face_coverage,
            # Face signal (new)
            e.mean_brow_furrow,
            e.mean_mouth_open,
            e.mean_eye_asymmetry,
            # Keystroke signal
            k.typing_speed_cps,
            k.mean_iki,
            k.std_iki,
            k.cv_iki,
            k.backspace_rate,
            k.burst_coefficient,
            float(k.pause_count),
        ], dtype=np.float32)

    FEATURE_NAMES = [
        # Eye
        "mean_ear", "std_ear", "blink_rate", "gaze_stability",
        "perclos", "head_roll_mean", "face_coverage",
        # Face (new)
        "brow_furrow_ratio", "mouth_open_ratio", "eye_asymmetry",
        # Keystroke
        "typing_speed_cps", "mean_iki", "std_iki", "cv_iki",
        "backspace_rate", "burst_coefficient", "pause_count",
    ]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERCLOS_EAR_CLOSED  = 0.21   # same as blink threshold
FAST_IKI_THRESHOLD  = 0.10   # seconds — "burst" typing
PAUSE_IKI_THRESHOLD = 1.00   # seconds — deliberate pause

# Face-signal reference baselines (used for normalisation in sliding window)
NORMAL_BFR  = 0.55   # relaxed brow furrow ratio
NORMAL_MAR  = 0.03   # resting mouth aspect ratio


# ---------------------------------------------------------------------------
# SlidingWindow
# ---------------------------------------------------------------------------
class SlidingWindow:
    """
    Drains eye and keystroke event queues, maintains rolling buffers,
    and periodically emits `WindowedFeatures` via a callback.

    Parameters
    ----------
    eye_queue      : queue.Queue[EyeFrame]        from EyeTracker
    keystroke_queue: queue.Queue[KeystrokeEvent]  from KeystrokeMonitor
    window_size    : float  — seconds of history to analyse  (default 7.0)
    step_size      : float  — seconds between successive windows (default 2.0)
    on_window      : Callable[[WindowedFeatures], None] — result callback
    """

    def __init__(
        self,
        eye_queue:       queue.Queue,
        keystroke_queue: queue.Queue,
        window_size:     float = 7.0,
        step_size:       float = 2.0,
        on_window:       Optional[Callable[[WindowedFeatures], None]] = None,
    ) -> None:
        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be > 0.")
        if step_size > window_size:
            raise ValueError("step_size must be ≤ window_size.")

        self.eye_queue       = eye_queue
        self.keystroke_queue = keystroke_queue
        self.window_size     = window_size
        self.step_size       = step_size
        self.on_window       = on_window or (lambda _: None)

        _max_eye = int(window_size * 35)
        _max_key = int(window_size * 30)
        self._eye_buf: Deque[EyeFrame]       = deque(maxlen=_max_eye)
        self._key_buf: Deque[KeystrokeEvent] = deque(maxlen=_max_key)

        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("SlidingWindow already running.")
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._aggregator_loop,
            daemon=True,
            name="SlidingWindowThread",
        )
        self._thread.start()
        logger.info(
            "SlidingWindow started (window=%.1fs, step=%.1fs).",
            self.window_size, self.step_size,
        )

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("SlidingWindow stopped.")

    # ------------------------------------------------------------------
    # Aggregator loop
    # ------------------------------------------------------------------
    def _aggregator_loop(self) -> None:
        next_trigger = time.monotonic() + self.step_size

        while self._running.is_set():
            now = time.monotonic()

            self._drain_queue(self.eye_queue,       self._eye_buf)
            self._drain_queue(self.keystroke_queue, self._key_buf)

            if now >= next_trigger:
                wall_now     = time.time()
                window_start = wall_now - self.window_size

                features = self._compute_window(window_start, wall_now)
                try:
                    self.on_window(features)
                except Exception:
                    logger.exception("on_window callback raised an error.")

                next_trigger = now + self.step_size

            time.sleep(0.02)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _drain_queue(q: queue.Queue, buf: deque) -> None:
        try:
            while True:
                buf.append(q.get_nowait())
        except queue.Empty:
            pass

    def _compute_window(
        self, window_start: float, window_end: float
    ) -> WindowedFeatures:
        eye_frames = [f for f in self._eye_buf if window_start <= f.timestamp <= window_end]
        key_events = [e for e in self._key_buf if window_start <= e.timestamp <= window_end]

        eye_stats = self._compute_eye_stats(eye_frames, window_start, window_end)
        key_stats = self._compute_key_stats(key_events, window_start, window_end)

        return WindowedFeatures(
            timestamp=window_end,
            eye_stats=eye_stats,
            key_stats=key_stats,
        )

    # ---- Eye + face stats ----
    def _compute_eye_stats(
        self,
        frames: List[EyeFrame],
        w_start: float,
        w_end:   float,
    ) -> EyeWindowStats:
        duration      = w_end - w_start
        n_total       = len(frames)
        face_frames   = [f for f in frames if f.face_detected]
        face_coverage = len(face_frames) / max(n_total, 1)

        if not face_frames:
            return EyeWindowStats(
                window_start=w_start, window_end=w_end,
                mean_ear=0.0, std_ear=0.0,
                blink_rate=0.0, gaze_stability=1.0,
                perclos=0.0, head_roll_mean=0.0,
                frames_captured=n_total, face_coverage=0.0,
                # face-signal defaults (neutral / relaxed)
                mean_brow_furrow=NORMAL_BFR,
                std_brow_furrow=0.0,
                mean_mouth_open=NORMAL_MAR,
                mean_eye_asymmetry=0.0,
            )

        ears = np.array([f.mean_ear for f in face_frames], dtype=np.float64)
        mean_ear = float(np.mean(ears))
        std_ear  = float(np.std(ears))

        # Blink rate
        blinks     = sum(1 for f in face_frames if f.blink_detected)
        minutes    = duration / 60.0
        blink_rate = blinks / max(minutes, 1e-6)

        # PERCLOS
        closed_frames = sum(1 for f in face_frames if f.mean_ear < PERCLOS_EAR_CLOSED)
        perclos       = closed_frames / max(len(face_frames), 1)

        # Gaze stability
        gaze_vecs = [f.gaze_vector for f in face_frames if f.gaze_vector is not None]
        if len(gaze_vecs) >= 2:
            gx = np.array([v[0] for v in gaze_vecs])
            gy = np.array([v[1] for v in gaze_vecs])
            gaze_var      = float(np.var(gx) + np.var(gy))
            gaze_stability = float(np.clip(1.0 - gaze_var / 0.01, 0.0, 1.0))
        else:
            gaze_stability = 1.0

        head_roll_mean = float(np.mean([f.head_roll for f in face_frames]))

        # ---- NEW: Face-signal aggregates ----
        bfr_arr   = np.array([f.brow_furrow_ratio for f in face_frames], dtype=np.float64)
        mar_arr   = np.array([f.mouth_open_ratio  for f in face_frames], dtype=np.float64)
        asym_arr  = np.array([f.eye_asymmetry     for f in face_frames], dtype=np.float64)

        mean_brow_furrow   = float(np.mean(bfr_arr))
        std_brow_furrow    = float(np.std(bfr_arr))
        mean_mouth_open    = float(np.mean(mar_arr))
        mean_eye_asymmetry = float(np.mean(asym_arr))

        return EyeWindowStats(
            window_start=w_start,
            window_end=w_end,
            mean_ear=mean_ear,
            std_ear=std_ear,
            blink_rate=blink_rate,
            gaze_stability=gaze_stability,
            perclos=perclos,
            head_roll_mean=head_roll_mean,
            frames_captured=n_total,
            face_coverage=face_coverage,
            mean_brow_furrow=mean_brow_furrow,
            std_brow_furrow=std_brow_furrow,
            mean_mouth_open=mean_mouth_open,
            mean_eye_asymmetry=mean_eye_asymmetry,
        )

    # ---- Keystroke stats ----
    def _compute_key_stats(
        self,
        events: List[KeystrokeEvent],
        w_start: float,
        w_end:   float,
    ) -> KeystrokeWindowStats:
        duration   = w_end - w_start
        press_evts = [e for e in events if e.event_type == "press"]
        total_ks   = len(press_evts)

        printable_ks = [e for e in press_evts if e.category == KeyCategory.PRINTABLE]
        backspace_ks = [e for e in press_evts if e.category == KeyCategory.BACKSPACE]

        typing_speed_cps = len(printable_ks) / max(duration, 1e-6)
        backspace_rate   = len(backspace_ks)  / max(total_ks, 1)

        ikis = [e.iki for e in press_evts if e.iki is not None and e.iki < self.window_size]

        if ikis:
            iki_arr           = np.array(ikis, dtype=np.float64)
            mean_iki          = float(np.mean(iki_arr))
            std_iki           = float(np.std(iki_arr))
            cv_iki            = std_iki / max(mean_iki, 1e-6)
            burst_coefficient = float(np.mean(iki_arr < FAST_IKI_THRESHOLD))
            pause_count       = int(np.sum(iki_arr > PAUSE_IKI_THRESHOLD))
        else:
            mean_iki = std_iki = cv_iki = burst_coefficient = 0.0
            pause_count = 0

        hold_times = [e.hold_time for e in events
                      if e.event_type == "release" and e.hold_time is not None]
        mean_hold = float(np.mean(hold_times)) if hold_times else 0.0

        return KeystrokeWindowStats(
            window_start=w_start,
            window_end=w_end,
            typing_speed_cps=typing_speed_cps,
            mean_iki=mean_iki,
            std_iki=std_iki,
            cv_iki=cv_iki,
            backspace_rate=backspace_rate,
            mean_hold=mean_hold,
            burst_coefficient=burst_coefficient,
            pause_count=pause_count,
            total_keystrokes=total_ks,
        )
