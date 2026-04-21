"""
acquisition/eye_tracker.py
==========================
Webcam-based eye AND face tracking using MediaPipe Face Mesh.

Computes per-frame:
  • Eye Aspect Ratio (EAR)           — blink detection
  • Blink rate                        — blinks / minute over rolling window
  • Gaze stability score              — variance of iris centre displacement
  • Head-pose deviation               — roll / yaw proxy
  • PERCLOS                           — Percentage of Eye CLOSure (drowsiness)
  • Brow Furrow Ratio (BFR)           — inner-brow distance / inter-ocular dist
                                        Low value → brows pulled together → stress
  • Mouth Open Ratio (MAR)            — vertical mouth / horizontal width
                                        Extreme (high OR low) signals tension/stress
  • Eye Asymmetry                     — |left_ear − right_ear|
                                        Elevated value signals facial tension / stress

Three-signal model
------------------
  Signal 1 – EAR / blink / gaze      (existing)
  Signal 2 – Face: brow + mouth + asymmetry  (NEW)
  Signal 3 – Keystroke dynamics       (in keystroke_monitor.py)

Threading model
---------------
EyeTracker runs its capture loop on a daemon thread so it never blocks
the main process on shutdown. Results are placed into a queue.Queue;
the SlidingWindow reads from it at its own pace.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (478-point Face Mesh with refine_landmarks=True)
# ---------------------------------------------------------------------------
# Right eye (from the subject's perspective)
RIGHT_EYE_INDICES = {
    "p1": 33,   # outer corner
    "p2": 160,  # upper lid – outer
    "p3": 158,  # upper lid – inner
    "p4": 133,  # inner corner
    "p5": 153,  # lower lid – inner
    "p6": 144,  # lower lid – outer
}
# Left eye
LEFT_EYE_INDICES = {
    "p1": 362,
    "p2": 385,
    "p3": 387,
    "p4": 263,
    "p5": 373,
    "p6": 380,
}
# Iris centres (only with refine_landmarks=True)
RIGHT_IRIS_CENTRE = 468
LEFT_IRIS_CENTRE  = 473

# Nose tip & chin for rough head-pose estimation
NOSE_TIP    = 1
CHIN        = 152
LEFT_EAR_L  = 234
RIGHT_EAR_L = 454

# ---- NEW: Brow furrow landmarks ----
# Inner brow corners (closest to the nose bridge)
BROW_INNER_LEFT  = 46    # left inner brow point
BROW_INNER_RIGHT = 276   # right inner brow point
# Outer eye corners used as the normalisation baseline (inter-ocular span)
OUTER_EYE_LEFT   = 362   # same as LEFT_EYE_INDICES["p1"]
OUTER_EYE_RIGHT  = 33    # same as RIGHT_EYE_INDICES["p1"]

# ---- NEW: Mouth landmarks (MAR) ----
MOUTH_LEFT  = 61   # left mouth corner
MOUTH_RIGHT = 291  # right mouth corner
MOUTH_UPPER = 13   # inner upper lip centre
MOUTH_LOWER = 14   # inner lower lip centre

# Blink detection threshold
EAR_BLINK_THRESHOLD  = 0.21
# Minimum consecutive frames with EAR < threshold to count as blink
BLINK_CONSEC_FRAMES  = 2
# PERCLOS: eye considered "closed" if EAR < this fraction of open baseline
PERCLOS_CLOSED_RATIO = 0.75


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EyeFrame:
    """
    One snapshot of eye + face features extracted from a single frame.

    New face-signal fields
    ----------------------
    brow_furrow_ratio : float
        Distance between inner brow corners / inter-ocular distance.
        Typical relaxed value ≈ 0.50–0.65. Drops toward 0.30 when
        brows are furrowed (concentration / stress).

    mouth_open_ratio : float
        Mouth Aspect Ratio = vertical mouth gap / mouth width.
        Near 0 = lips pressed together (jaw clenching).
        > 0.25 = significantly open (fatigue / surprise).

    eye_asymmetry : float
        |left_ear − right_ear|.  > 0.05 suggests facial tension or
        asymmetric muscle activation common under stress.
    """
    timestamp:          float
    left_ear:           float
    right_ear:          float
    mean_ear:           float
    blink_detected:     bool
    iris_left:          Optional[tuple]
    iris_right:         Optional[tuple]
    gaze_vector:        Optional[tuple]
    head_roll:          float
    face_detected:      bool
    # ---- new face-signal fields ----
    brow_furrow_ratio:  float = 0.55   # default: relaxed
    mouth_open_ratio:   float = 0.03   # default: lightly closed
    eye_asymmetry:      float = 0.0


@dataclass
class EyeWindowStats:
    """Aggregated statistics over a sliding window of EyeFrame objects."""
    window_start:        float
    window_end:          float
    mean_ear:            float
    std_ear:             float
    blink_rate:          float   # blinks / minute
    gaze_stability:      float   # 1 – normalised variance (higher = more stable)
    perclos:             float   # fraction of frames where eye was 'closed'
    head_roll_mean:      float
    frames_captured:     int
    face_coverage:       float   # fraction of frames where face was detected
    # ---- new face-signal aggregates ----
    mean_brow_furrow:    float = 0.55   # mean BFR over window
    std_brow_furrow:     float = 0.0    # BFR variability
    mean_mouth_open:     float = 0.03   # mean MAR over window
    mean_eye_asymmetry:  float = 0.0    # mean |L-R EAR| over window


# ---------------------------------------------------------------------------
# EAR helper (pure function, easy to unit-test)
# ---------------------------------------------------------------------------
def compute_ear(landmarks, eye_indices: dict, img_w: int, img_h: int) -> float:
    """
    Eye Aspect Ratio as defined by Soukupová & Čech (2016):

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Parameters
    ----------
    landmarks  : MediaPipe NormalizedLandmarkList
    eye_indices: dict with keys p1…p6 mapping to landmark indices
    img_w, img_h: frame dimensions for denormalisation

    Returns
    -------
    float – EAR value (typically 0.25–0.45 when open, <0.21 when closed)
    """
    def lm(key: str) -> np.ndarray:
        lk = landmarks[eye_indices[key]]
        return np.array([lk.x * img_w, lk.y * img_h], dtype=np.float64)

    p1, p2, p3 = lm("p1"), lm("p2"), lm("p3")
    p4, p5, p6 = lm("p4"), lm("p5"), lm("p6")

    numerator   = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    denominator = 2.0 * np.linalg.norm(p1 - p4)

    if denominator < 1e-6:
        return 0.0
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# NEW: Brow Furrow Ratio (BFR)
# ---------------------------------------------------------------------------
def compute_brow_furrow_ratio(
    landmarks, img_w: int, img_h: int
) -> float:
    """
    Brow Furrow Ratio = distance(inner_brow_left, inner_brow_right)
                        / distance(outer_eye_left, outer_eye_right)

    Normalising by inter-ocular distance makes it camera-distance–invariant.

    Returns
    -------
    float typically in [0.30, 0.70].
    Lower → more furrowed (stressed / concentrating).
    """
    def pt(idx: int) -> np.ndarray:
        lk = landmarks[idx]
        return np.array([lk.x * img_w, lk.y * img_h], dtype=np.float64)

    inner_left   = pt(BROW_INNER_LEFT)
    inner_right  = pt(BROW_INNER_RIGHT)
    outer_left   = pt(OUTER_EYE_LEFT)
    outer_right  = pt(OUTER_EYE_RIGHT)

    brow_dist    = float(np.linalg.norm(inner_left  - inner_right))
    interocular  = float(np.linalg.norm(outer_left  - outer_right))

    if interocular < 1e-6:
        return 0.55   # fallback neutral
    return brow_dist / interocular


# ---------------------------------------------------------------------------
# NEW: Mouth Open Ratio (MAR)
# ---------------------------------------------------------------------------
def compute_mouth_open_ratio(
    landmarks, img_w: int, img_h: int
) -> float:
    """
    Mouth Aspect Ratio = ||upper_lip - lower_lip|| / ||left_corner - right_corner||

    Returns
    -------
    float typically in [0.0, 0.35].
    Near 0    → lips tightly compressed (jaw clenching under stress).
    0.02–0.08 → normal resting position.
    > 0.20    → significantly open (fatigue, surprise, deep breathing).
    """
    def pt(idx: int) -> np.ndarray:
        lk = landmarks[idx]
        return np.array([lk.x * img_w, lk.y * img_h], dtype=np.float64)

    left_corner  = pt(MOUTH_LEFT)
    right_corner = pt(MOUTH_RIGHT)
    upper_lip    = pt(MOUTH_UPPER)
    lower_lip    = pt(MOUTH_LOWER)

    vertical     = float(np.linalg.norm(upper_lip  - lower_lip))
    horizontal   = float(np.linalg.norm(left_corner - right_corner))

    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# EyeTracker
# ---------------------------------------------------------------------------
class EyeTracker:
    """
    Background thread that captures webcam frames, runs MediaPipe Face Mesh,
    and pushes `EyeFrame` objects (now including face-signal features) into
    an output queue.

    Signal coverage
    ---------------
    • Eye signal  : EAR, blink rate, gaze stability, PERCLOS
    • Face signal : brow furrow ratio, mouth open ratio, eye asymmetry

    Usage
    -----
    >>> tracker = EyeTracker(camera_index=0, target_fps=30)
    >>> tracker.start()
    >>> frame = tracker.frame_queue.get()
    >>> tracker.stop()
    """

    def __init__(
        self,
        camera_index:  int  = 0,
        target_fps:    int  = 30,
        queue_maxsize: int  = 128,
        show_preview:  bool = False,
    ) -> None:
        self.camera_index  = camera_index
        self.target_fps    = target_fps
        self.show_preview  = show_preview

        self.frame_queue: queue.Queue[EyeFrame] = queue.Queue(maxsize=queue_maxsize)

        self._running          = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._blink_counter    = 0
        self._total_blinks     = 0
        self._ear_baseline     = 0.30
        self._baseline_samples = deque(maxlen=300)

        self._mp_face_mesh = None
        self._face_mesh    = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("EyeTracker already running.")
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="EyeTrackerThread"
        )
        self._thread.start()
        logger.info("EyeTracker started (camera_index=%d, fps=%d).",
                    self.camera_index, self.target_fps)

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self.show_preview:
            cv2.destroyAllWindows()
        logger.info("EyeTracker stopped.")

    # ------------------------------------------------------------------
    # Private – capture loop
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error("Cannot open camera index %d.", self.camera_index)
            self._running.clear()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        frame_interval = 1.0 / self.target_fps

        try:
            while self._running.is_set():
                loop_start = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame – skipping.")
                    time.sleep(0.01)
                    continue

                eye_frame = self._process_frame(frame)

                try:
                    self.frame_queue.put_nowait(eye_frame)
                except queue.Full:
                    pass

                if self.show_preview:
                    self._draw_overlay(frame, eye_frame)
                    cv2.imshow("CognitiveMonitor – Eye+Face Tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                elapsed  = time.monotonic() - loop_start
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            cap.release()
            self._face_mesh.close()

    def _process_frame(self, frame: np.ndarray) -> EyeFrame:
        """Run MediaPipe on one BGR frame and return an EyeFrame."""
        timestamp = time.time()
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return EyeFrame(
                timestamp=timestamp,
                left_ear=0.0, right_ear=0.0, mean_ear=0.0,
                blink_detected=False,
                iris_left=None, iris_right=None,
                gaze_vector=None, head_roll=0.0,
                face_detected=False,
                brow_furrow_ratio=0.55,
                mouth_open_ratio=0.03,
                eye_asymmetry=0.0,
            )

        lms = results.multi_face_landmarks[0].landmark

        # ---- EAR ----
        left_ear  = compute_ear(lms, LEFT_EYE_INDICES,  w, h)
        right_ear = compute_ear(lms, RIGHT_EYE_INDICES, w, h)
        mean_ear  = (left_ear + right_ear) / 2.0

        # Adaptive baseline (open-eye frames only)
        if mean_ear > EAR_BLINK_THRESHOLD:
            self._baseline_samples.append(mean_ear)
            if len(self._baseline_samples) >= 30:
                self._ear_baseline = float(np.median(self._baseline_samples))

        # ---- Blink detection ----
        blink_detected = False
        if mean_ear < EAR_BLINK_THRESHOLD:
            self._blink_counter += 1
        else:
            if self._blink_counter >= BLINK_CONSEC_FRAMES:
                self._total_blinks += 1
                blink_detected = True
            self._blink_counter = 0

        # ---- Iris positions ----
        def iris_xy(idx: int) -> tuple:
            lm = lms[idx]
            return (lm.x, lm.y)

        iris_left  = iris_xy(LEFT_IRIS_CENTRE)
        iris_right = iris_xy(RIGHT_IRIS_CENTRE)

        # Gaze vector: iris displacement from eye-corner midpoint
        left_outer = np.array([lms[LEFT_EYE_INDICES["p1"]].x, lms[LEFT_EYE_INDICES["p1"]].y])
        left_inner = np.array([lms[LEFT_EYE_INDICES["p4"]].x, lms[LEFT_EYE_INDICES["p4"]].y])
        left_mid   = (left_outer + left_inner) / 2.0
        gaze_dx = iris_left[0] - float(left_mid[0])
        gaze_dy = iris_left[1] - float(left_mid[1])

        # ---- Head roll ----
        left_corner  = np.array([lms[LEFT_EYE_INDICES["p1"]].x  * w, lms[LEFT_EYE_INDICES["p1"]].y  * h])
        right_corner = np.array([lms[RIGHT_EYE_INDICES["p1"]].x * w, lms[RIGHT_EYE_INDICES["p1"]].y * h])
        delta     = right_corner - left_corner
        head_roll = math.degrees(math.atan2(delta[1], delta[0]))

        # ---- NEW: Face signal features ----
        brow_furrow_ratio = compute_brow_furrow_ratio(lms, w, h)
        mouth_open_ratio  = compute_mouth_open_ratio(lms, w, h)
        eye_asymmetry     = abs(left_ear - right_ear)

        return EyeFrame(
            timestamp=timestamp,
            left_ear=left_ear,
            right_ear=right_ear,
            mean_ear=mean_ear,
            blink_detected=blink_detected,
            iris_left=iris_left,
            iris_right=iris_right,
            gaze_vector=(gaze_dx, gaze_dy),
            head_roll=head_roll,
            face_detected=True,
            brow_furrow_ratio=brow_furrow_ratio,
            mouth_open_ratio=mouth_open_ratio,
            eye_asymmetry=eye_asymmetry,
        )

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------
    def _draw_overlay(self, frame: np.ndarray, ef: EyeFrame) -> None:
        color = (0, 200, 0) if ef.face_detected else (0, 0, 200)
        cv2.putText(frame, f"EAR: {ef.mean_ear:.3f}",          (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Blinks: {self._total_blinks}",    (10, 55),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"BFR: {ef.brow_furrow_ratio:.3f}", (10, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"MAR: {ef.mouth_open_ratio:.3f}",  (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Asym: {ef.eye_asymmetry:.3f}",    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if ef.blink_detected:
            cv2.putText(frame, "BLINK", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
