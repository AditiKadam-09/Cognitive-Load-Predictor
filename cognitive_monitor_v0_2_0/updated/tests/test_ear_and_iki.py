"""
tests/test_ear_and_iki.py
==========================
Unit tests for the three-signal cognitive-load model:
  Signal 1 – Eye  : EAR calculation, blink detection
  Signal 2 – Face : Brow Furrow Ratio (BFR), Mouth Open Ratio (MAR), eye asymmetry
  Signal 3 – Keys : IKI timing, window statistics

Test classes
------------
1. TestComputeEAR          — EAR formula correctness
2. TestFaceSignal          — NEW: BFR and MAR pure functions
3. TestCategoriseKey       — key category classification
4. TestKeystrokeWindowStats — IKI / typing stats
5. TestHeuristicScorer     — score thresholds with 17-feature vectors
6. TestHighLoadSimulation  — end-to-end high-load injection

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import math
import queue
import time
import types
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Mock pynput for headless/CI environments
# ---------------------------------------------------------------------------
import sys
from unittest.mock import MagicMock


def _mock_pynput():
    from enum import Enum, auto

    class Key(Enum):
        backspace = auto(); delete = auto()
        shift = auto(); shift_r = auto()
        ctrl = auto(); ctrl_r = auto()
        alt = auto(); alt_r = auto(); alt_gr = auto()
        cmd = auto(); cmd_r = auto()
        up = auto(); down = auto(); left = auto(); right = auto()
        home = auto(); end = auto(); page_up = auto(); page_down = auto()
        enter = auto(); escape = auto()

    class KeyCode:
        def __init__(self, vk=None, char=None):
            self.vk   = vk
            self.char = char

        @classmethod
        def from_char(cls, c):
            return cls(vk=ord(c), char=c)

    class Listener:
        running = False
        daemon  = True
        def __init__(self, on_press=None, on_release=None): pass
        def start(self): self.running = True
        def stop(self):  self.running = False
        def join(self, **kw): pass

    kbd_mock          = MagicMock()
    kbd_mock.Key      = Key
    kbd_mock.KeyCode  = KeyCode
    kbd_mock.Listener = Listener

    pynput_mock          = MagicMock()
    pynput_mock.keyboard = kbd_mock
    sys.modules["pynput"]          = pynput_mock
    sys.modules["pynput.keyboard"] = kbd_mock


_mock_pynput()

# ---------------------------------------------------------------------------
# Mock mediapipe so tests run without a GPU
# ---------------------------------------------------------------------------
mp_mock = MagicMock()
sys.modules.setdefault("mediapipe", mp_mock)


# ---------------------------------------------------------------------------
# Minimal landmark mock
# ---------------------------------------------------------------------------
class _LandmarkMock:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class _LandmarkListMock:
    def __init__(self, mapping: dict):
        self._m = mapping

    def __getitem__(self, idx: int) -> _LandmarkMock:
        return self._m[idx]


# ---------------------------------------------------------------------------
# Module imports (after mocks)
# ---------------------------------------------------------------------------
from cognitive_monitor.acquisition.eye_tracker import (
    compute_ear,
    compute_brow_furrow_ratio,
    compute_mouth_open_ratio,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    EAR_BLINK_THRESHOLD,
    BROW_INNER_LEFT, BROW_INNER_RIGHT,
    OUTER_EYE_LEFT, OUTER_EYE_RIGHT,
    MOUTH_LEFT, MOUTH_RIGHT, MOUTH_UPPER, MOUTH_LOWER,
)
from cognitive_monitor.acquisition.keystroke_monitor import (
    KeystrokeEvent,
    KeystrokeWindowStats,
    KeyCategory,
    categorise_key,
)
from cognitive_monitor.acquisition.sliding_window import (
    SlidingWindow,
    WindowedFeatures,
    FAST_IKI_THRESHOLD,
    PAUSE_IKI_THRESHOLD,
)
from cognitive_monitor.analysis.feature_extractor import (
    HeuristicScorer,
    FeatureExtractor,
    CognitiveLoadClass,
    score_to_class,
)


# ===========================================================================
# 1. EAR Calculation Tests (unchanged)
# ===========================================================================
class TestComputeEAR(unittest.TestCase):

    IMG_W = 640
    IMG_H = 480

    def _build_landmarks(self, p1, p2, p3, p4, p5, p6, indices) -> _LandmarkListMock:
        pts  = [p1, p2, p3, p4, p5, p6]
        keys = ["p1", "p2", "p3", "p4", "p5", "p6"]
        mapping = {}
        for key, (px, py) in zip(keys, pts):
            idx = indices[key]
            mapping[idx] = _LandmarkMock(px / self.IMG_W, py / self.IMG_H)
        return _LandmarkListMock(mapping)

    def test_open_eye_ear_typical_range(self):
        lm = self._build_landmarks(
            p1=(100, 240), p2=(113, 230), p3=(127, 229),
            p4=(140, 240), p5=(127, 251), p6=(113, 252),
            indices=LEFT_EYE_INDICES,
        )
        ear = compute_ear(lm, LEFT_EYE_INDICES, self.IMG_W, self.IMG_H)
        self.assertGreater(ear, 0.20)
        self.assertLess(ear,    0.70)

    def test_closed_eye_ear_below_threshold(self):
        lm = self._build_landmarks(
            p1=(100, 240), p2=(110, 240), p3=(120, 240),
            p4=(140, 240), p5=(130, 240), p6=(110, 240),
            indices=LEFT_EYE_INDICES,
        )
        ear = compute_ear(lm, LEFT_EYE_INDICES, self.IMG_W, self.IMG_H)
        self.assertLess(ear, EAR_BLINK_THRESHOLD)

    def test_ear_formula_manual_calculation(self):
        lm = self._build_landmarks(
            p1=(0, 0), p2=(1, 2), p3=(2, 3),
            p4=(6, 0), p5=(4, 3), p6=(5, 2),
            indices=LEFT_EYE_INDICES,
        )
        ear      = compute_ear(lm, LEFT_EYE_INDICES, 1, 1)
        expected = (
            math.dist((1, 2), (5, 2)) + math.dist((2, 3), (4, 3))
        ) / (2 * math.dist((0, 0), (6, 0)))
        self.assertAlmostEqual(ear, expected, places=5)

    def test_degenerate_zero_width_eye(self):
        lm = self._build_landmarks(
            p1=(100, 240), p2=(101, 238), p3=(102, 237),
            p4=(100, 240), p5=(101, 242), p6=(100, 241),
            indices=LEFT_EYE_INDICES,
        )
        ear = compute_ear(lm, LEFT_EYE_INDICES, self.IMG_W, self.IMG_H)
        self.assertEqual(ear, 0.0)

    def test_right_eye_indices_used(self):
        lm = self._build_landmarks(
            p1=(400, 240), p2=(410, 230), p3=(420, 228),
            p4=(440, 240), p5=(430, 252), p6=(410, 252),
            indices=RIGHT_EYE_INDICES,
        )
        ear = compute_ear(lm, RIGHT_EYE_INDICES, self.IMG_W, self.IMG_H)
        self.assertGreater(ear, 0.0)


# ===========================================================================
# 2. Face Signal Tests (NEW)
# ===========================================================================
class TestFaceSignal(unittest.TestCase):
    """Tests for compute_brow_furrow_ratio and compute_mouth_open_ratio."""

    IMG_W = 640
    IMG_H = 480

    def _landmark_list(self, mapping: dict) -> _LandmarkListMock:
        """Build a landmark list from {index: (pixel_x, pixel_y)}."""
        norm = {
            idx: _LandmarkMock(px / self.IMG_W, py / self.IMG_H)
            for idx, (px, py) in mapping.items()
        }
        return _LandmarkListMock(norm)

    # ---- Brow Furrow Ratio ----

    def test_bfr_relaxed_range(self):
        """Relaxed brows: inner brow points are well apart → BFR ≈ 0.5–0.70."""
        # Inner brows at x=260 and x=380 → distance = 120 px
        # Outer eye corners at x=160 and x=480 → inter-ocular = 320 px
        # BFR = 120/320 = 0.375 ... relaxed real values are wider; let's use bigger separation
        # Inner brows 220/420 → distance=200, inter-ocular 160/480=320 → BFR=0.625
        lm = self._landmark_list({
            BROW_INNER_LEFT:  (220, 200),
            BROW_INNER_RIGHT: (420, 200),
            OUTER_EYE_LEFT:   (160, 240),
            OUTER_EYE_RIGHT:  (480, 240),
        })
        bfr = compute_brow_furrow_ratio(lm, self.IMG_W, self.IMG_H)
        self.assertGreater(bfr, 0.50, f"Relaxed BFR too low: {bfr:.3f}")
        self.assertLess(bfr,    0.80, f"Relaxed BFR too high: {bfr:.3f}")

    def test_bfr_furrowed_is_lower_than_relaxed(self):
        """Furrowed brows: inner points closer together → lower BFR."""
        # Same inter-ocular baseline; inner brows drawn inward
        relaxed_lm = self._landmark_list({
            BROW_INNER_LEFT:  (220, 200),
            BROW_INNER_RIGHT: (420, 200),
            OUTER_EYE_LEFT:   (160, 240),
            OUTER_EYE_RIGHT:  (480, 240),
        })
        furrowed_lm = self._landmark_list({
            BROW_INNER_LEFT:  (295, 200),   # moved inward by 75 px
            BROW_INNER_RIGHT: (345, 200),   # moved inward by 75 px
            OUTER_EYE_LEFT:   (160, 240),
            OUTER_EYE_RIGHT:  (480, 240),
        })
        bfr_relaxed  = compute_brow_furrow_ratio(relaxed_lm,  self.IMG_W, self.IMG_H)
        bfr_furrowed = compute_brow_furrow_ratio(furrowed_lm, self.IMG_W, self.IMG_H)
        self.assertGreater(
            bfr_relaxed, bfr_furrowed,
            f"Relaxed BFR ({bfr_relaxed:.3f}) should exceed furrowed ({bfr_furrowed:.3f})"
        )

    def test_bfr_degenerate_zero_interocular(self):
        """Outer eye corners at same point → should not crash, returns fallback."""
        lm = self._landmark_list({
            BROW_INNER_LEFT:  (320, 200),
            BROW_INNER_RIGHT: (330, 200),
            OUTER_EYE_LEFT:   (320, 240),   # same x as right
            OUTER_EYE_RIGHT:  (320, 240),
        })
        bfr = compute_brow_furrow_ratio(lm, self.IMG_W, self.IMG_H)
        self.assertAlmostEqual(bfr, 0.55, places=2)   # neutral fallback

    # ---- Mouth Open Ratio ----

    def test_mar_closed_mouth_near_zero(self):
        """Upper and lower lip at same y → MAR ≈ 0."""
        lm = self._landmark_list({
            MOUTH_LEFT:  (260, 360),
            MOUTH_RIGHT: (380, 360),
            MOUTH_UPPER: (320, 355),
            MOUTH_LOWER: (320, 355),   # same y → vertical dist = 0
        })
        mar = compute_mouth_open_ratio(lm, self.IMG_W, self.IMG_H)
        self.assertAlmostEqual(mar, 0.0, places=3)

    def test_mar_open_mouth(self):
        """Wide-open mouth: large vertical / horizontal ratio."""
        # Width = 120 px, vertical gap = 40 px → MAR ≈ 0.33
        lm = self._landmark_list({
            MOUTH_LEFT:  (260, 360),
            MOUTH_RIGHT: (380, 360),
            MOUTH_UPPER: (320, 345),
            MOUTH_LOWER: (320, 385),
        })
        mar = compute_mouth_open_ratio(lm, self.IMG_W, self.IMG_H)
        expected = 40 / 120  # ≈ 0.333
        self.assertAlmostEqual(mar, expected, places=3)

    def test_mar_typical_resting_range(self):
        """Typical resting MAR is small but positive."""
        # Width 120 px, 4 px gap → MAR ≈ 0.033
        lm = self._landmark_list({
            MOUTH_LEFT:  (260, 360),
            MOUTH_RIGHT: (380, 360),
            MOUTH_UPPER: (320, 358),
            MOUTH_LOWER: (320, 362),
        })
        mar = compute_mouth_open_ratio(lm, self.IMG_W, self.IMG_H)
        self.assertGreater(mar, 0.0)
        self.assertLess(mar,    0.10)

    def test_mar_degenerate_zero_width(self):
        """Zero horizontal mouth width → should return 0.0, not crash."""
        lm = self._landmark_list({
            MOUTH_LEFT:  (320, 360),
            MOUTH_RIGHT: (320, 360),   # same x → width = 0
            MOUTH_UPPER: (320, 355),
            MOUTH_LOWER: (320, 365),
        })
        mar = compute_mouth_open_ratio(lm, self.IMG_W, self.IMG_H)
        self.assertEqual(mar, 0.0)


# ===========================================================================
# 3. Key Categorisation Tests (unchanged)
# ===========================================================================
import pynput.keyboard as _kb


class TestCategoriseKey(unittest.TestCase):

    def test_backspace_classified(self):
        self.assertEqual(categorise_key(_kb.Key.backspace), KeyCategory.BACKSPACE)

    def test_delete_classified(self):
        self.assertEqual(categorise_key(_kb.Key.delete), KeyCategory.BACKSPACE)

    def test_shift_is_modifier(self):
        self.assertEqual(categorise_key(_kb.Key.shift), KeyCategory.MODIFIER)

    def test_arrow_is_navigation(self):
        self.assertEqual(categorise_key(_kb.Key.up), KeyCategory.NAVIGATION)

    def test_printable_char(self):
        key = _kb.KeyCode.from_char("a")
        self.assertEqual(categorise_key(key), KeyCategory.PRINTABLE)

    def test_digit_is_printable(self):
        key = _kb.KeyCode.from_char("5")
        self.assertEqual(categorise_key(key), KeyCategory.PRINTABLE)

    def test_enter_is_other(self):
        self.assertEqual(categorise_key(_kb.Key.enter), KeyCategory.OTHER)


# ===========================================================================
# 4. IKI / Keystroke Window Stats (unchanged logic, verifies stats)
# ===========================================================================
class TestKeystrokeWindowStats(unittest.TestCase):

    def _make_window(self):
        return SlidingWindow(
            eye_queue=queue.Queue(),
            keystroke_queue=queue.Queue(),
            window_size=7.0,
            step_size=2.0,
        )

    def _make_press(self, ts, iki, cat=KeyCategory.PRINTABLE):
        return KeystrokeEvent(
            timestamp=ts, event_type="press",
            category=cat, key_id=ord("a"), iki=iki,
        )

    def test_typing_speed_calculated(self):
        sw  = self._make_window()
        now = 1000.0
        events = [self._make_press(now + i * 0.15, 0.15) for i in range(20)]
        stats  = sw._compute_key_stats(events, now, now + 7.0)
        self.assertAlmostEqual(stats.typing_speed_cps, 20 / 7.0, places=2)

    def test_backspace_rate(self):
        sw  = self._make_window()
        now = 2000.0
        regular  = [self._make_press(now + i * 0.2, 0.2) for i in range(9)]
        backsps  = [self._make_press(now + 1.8 + i * 0.2, 0.2, KeyCategory.BACKSPACE) for i in range(1)]
        stats    = sw._compute_key_stats(regular + backsps, now, now + 7.0)
        self.assertAlmostEqual(stats.backspace_rate, 1 / 10, places=3)

    def test_high_backspace_rate(self):
        sw  = self._make_window()
        now = 3000.0
        regular = [self._make_press(now + i * 0.1, 0.1) for i in range(10)]
        backsps = [self._make_press(now + 1.0 + i * 0.1, 0.1, KeyCategory.BACKSPACE) for i in range(10)]
        stats   = sw._compute_key_stats(regular + backsps, now, now + 7.0)
        self.assertGreater(stats.backspace_rate, 0.45)

    def test_burst_coefficient(self):
        sw      = self._make_window()
        now     = 4000.0
        fast_iki = FAST_IKI_THRESHOLD * 0.5
        events  = [self._make_press(now + i * fast_iki, fast_iki) for i in range(30)]
        stats   = sw._compute_key_stats(events, now, now + 7.0)
        self.assertGreater(stats.burst_coefficient, 0.90)

    def test_pause_count(self):
        sw  = self._make_window()
        now = 5000.0
        events = []
        for i in range(5):
            events.append(self._make_press(now + i * 0.15, 0.15))
        events.append(self._make_press(now + 5 * 0.15 + 2.0, 2.0))
        for i in range(5):
            events.append(self._make_press(now + 5 * 0.15 + 2.0 + i * 0.15, 0.15))
        events.append(self._make_press(now + 5 * 0.15 + 2.0 + 5 * 0.15 + 1.5, 1.5))
        stats = sw._compute_key_stats(events, now, now + 7.0)
        self.assertEqual(stats.pause_count, 2)

    def test_empty_events(self):
        sw    = self._make_window()
        stats = sw._compute_key_stats([], 0.0, 7.0)
        self.assertEqual(stats.typing_speed_cps, 0.0)
        self.assertEqual(stats.mean_iki,         0.0)
        self.assertEqual(stats.total_keystrokes, 0)


# ===========================================================================
# 5. Heuristic Scorer — 17-feature vectors
# ===========================================================================
class TestHeuristicScorer(unittest.TestCase):

    def _relaxed_fv(self) -> np.ndarray:
        """17-feature vector: calm, rested user — all three signals normal."""
        return np.array([
            # Eye signal
            0.32,   # mean_ear          (normal open)
            0.02,   # std_ear
            14.0,   # blink_rate        (normal)
            0.95,   # gaze_stability    (very stable)
            0.02,   # perclos
            1.0,    # head_roll_mean
            0.98,   # face_coverage
            # Face signal (new)
            0.60,   # brow_furrow_ratio (relaxed, wide apart)
            0.03,   # mouth_open_ratio  (resting)
            0.01,   # eye_asymmetry     (minimal)
            # Keystroke signal
            3.0,    # typing_speed_cps
            0.18,   # mean_iki
            0.03,   # std_iki
            0.17,   # cv_iki
            0.04,   # backspace_rate
            0.05,   # burst_coefficient
            0.0,    # pause_count
        ], dtype=np.float32)

    def _stressed_fv(self) -> np.ndarray:
        """17-feature vector: highly stressed user — all three signals abnormal."""
        return np.array([
            # Eye signal
            0.18,   # mean_ear          (near-closed)
            0.06,   # std_ear
            5.0,    # blink_rate        (very low → concentration stress)
            0.30,   # gaze_stability    (erratic)
            0.35,   # perclos           (35 % of time closed)
            8.0,    # head_roll_mean
            0.80,   # face_coverage
            # Face signal (new) — all stress indicators active
            0.32,   # brow_furrow_ratio (furrowed brows — low value)
            0.01,   # mouth_open_ratio  (jaw clenching — very tight)
            0.10,   # eye_asymmetry     (high facial tension)
            # Keystroke signal
            6.0,    # typing_speed_cps  (fast / frantic)
            0.07,   # mean_iki          (very fast)
            0.08,   # std_iki
            1.14,   # cv_iki            (very irregular)
            0.30,   # backspace_rate    (high error rate)
            0.80,   # burst_coefficient
            5.0,    # pause_count
        ], dtype=np.float32)

    def test_relaxed_state_is_low(self):
        scorer = HeuristicScorer()
        score  = scorer.score(self._relaxed_fv())
        self.assertLess(score, 40, f"Relaxed score too high: {score:.1f}")

    def test_stressed_state_is_high(self):
        scorer = HeuristicScorer()
        score  = scorer.score(self._stressed_fv())
        self.assertGreater(score, 60, f"Stressed score too low: {score:.1f}")

    def test_furrowed_brows_raises_score(self):
        """Furrowed brows alone should increase score versus relaxed brows."""
        scorer     = HeuristicScorer()
        normal_fv  = self._relaxed_fv()
        furrowed_fv = normal_fv.copy()
        furrowed_fv[7] = 0.32   # low BFR = furrowed
        self.assertGreater(
            scorer.score(furrowed_fv),
            scorer.score(normal_fv),
            "Furrowed brows should produce a higher load score.",
        )

    def test_jaw_clenching_raises_score(self):
        """Very tight mouth (jaw clenching) should increase score."""
        scorer    = HeuristicScorer()
        normal_fv = self._relaxed_fv()
        clench_fv = normal_fv.copy()
        clench_fv[8] = 0.005   # near-zero MAR = jaw clenching
        self.assertGreater(
            scorer.score(clench_fv),
            scorer.score(normal_fv),
            "Jaw clenching should produce a higher load score.",
        )

    def test_wide_open_mouth_raises_score(self):
        """Wide-open mouth (fatigue/gasp) should also increase score."""
        scorer    = HeuristicScorer()
        normal_fv = self._relaxed_fv()
        gape_fv   = normal_fv.copy()
        gape_fv[8] = 0.25   # very open mouth
        self.assertGreater(
            scorer.score(gape_fv),
            scorer.score(normal_fv),
            "Wide-open mouth should produce a higher load score.",
        )

    def test_eye_asymmetry_raises_score(self):
        """High eye asymmetry should increase score."""
        scorer    = HeuristicScorer()
        normal_fv = self._relaxed_fv()
        asym_fv   = normal_fv.copy()
        asym_fv[9] = 0.12   # high asymmetry
        self.assertGreater(
            scorer.score(asym_fv),
            scorer.score(normal_fv),
            "Eye asymmetry should produce a higher load score.",
        )

    def test_score_to_class_mapping(self):
        self.assertEqual(score_to_class(10),  CognitiveLoadClass.LOW)
        self.assertEqual(score_to_class(45),  CognitiveLoadClass.MEDIUM)
        self.assertEqual(score_to_class(70),  CognitiveLoadClass.HIGH)
        self.assertEqual(score_to_class(85),  CognitiveLoadClass.CRITICAL)
        self.assertEqual(score_to_class(100), CognitiveLoadClass.CRITICAL)

    def test_intervention_message_high(self):
        extractor = FeatureExtractor()
        extractor._heuristic.score = lambda fv: 65.0
        wf = MagicMock()
        wf.feature_vector = self._stressed_fv()
        wf.timestamp      = time.time()
        result = extractor.predict(wf)
        self.assertEqual(result.load_class, CognitiveLoadClass.HIGH)
        self.assertTrue(result.needs_intervention)
        self.assertIn("breathing", result.intervention_message)

    def test_intervention_message_critical(self):
        extractor = FeatureExtractor()
        extractor._heuristic.score = lambda fv: 90.0
        wf = MagicMock()
        wf.feature_vector = self._stressed_fv()
        wf.timestamp      = time.time()
        result = extractor.predict(wf)
        self.assertEqual(result.load_class, CognitiveLoadClass.CRITICAL)
        self.assertIn("Step away", result.intervention_message)

    def test_feature_vector_length(self):
        """feature_vector property must return exactly 17 elements."""
        sw = SlidingWindow(
            eye_queue=queue.Queue(),
            keystroke_queue=queue.Queue(),
            window_size=7.0,
            step_size=2.0,
        )
        now = time.time()
        wf  = sw._compute_window(now - 7.0, now)   # empty queues
        self.assertEqual(len(wf.feature_vector), 17,
                         "feature_vector must have 17 elements")


# ===========================================================================
# 6. Functional Test — Simulate "High Load" (three-signal)
# ===========================================================================
class TestHighLoadSimulation(unittest.TestCase):
    """
    Injects synthetic high-load events for all three signals:
    • Eye  : near-closed EAR, erratic gaze
    • Face : furrowed brows (low BFR), tight mouth (low MAR), high asymmetry
    • Keys : rapid bursts with heavy backspace usage
    """

    def test_high_load_triggers_alert(self):
        now         = time.time()
        eye_q       = queue.Queue()
        keystroke_q = queue.Queue()

        from cognitive_monitor.acquisition.eye_tracker import EyeFrame

        # ---- 200 frames: near-closed eyes, furrowed brows, tight jaw ----
        for i in range(200):
            eye_q.put(EyeFrame(
                timestamp          = now - 7.0 + i * 0.035,
                left_ear           = 0.16 + 0.02 * math.sin(i),
                right_ear          = 0.18 + 0.02 * math.cos(i),   # asymmetric
                mean_ear           = 0.17,
                blink_detected     = (i % 20 == 0),
                iris_left          = (0.5 + 0.04 * math.sin(i * 0.8), 0.5),
                iris_right         = (0.5 + 0.04 * math.cos(i * 0.8), 0.5),
                gaze_vector        = (0.04 * math.sin(i), 0.04 * math.cos(i)),
                head_roll          = 5.0 * math.sin(i * 0.3),
                face_detected      = True,
                # Face signal: furrowed + clenching + asymmetric
                brow_furrow_ratio  = 0.32 + 0.02 * math.sin(i * 0.5),
                mouth_open_ratio   = 0.005 + 0.003 * abs(math.sin(i)),
                eye_asymmetry      = abs(0.16 - 0.18) + 0.02 * abs(math.sin(i)),
            ))

        # ---- 60 fast keystrokes, 33 % backspace ----
        t = now - 7.0
        for i in range(60):
            cat = KeyCategory.BACKSPACE if (i % 3 == 0) else KeyCategory.PRINTABLE
            iki = 0.06 if (i % 5 != 0) else 0.03
            keystroke_q.put(KeystrokeEvent(
                timestamp  = t,
                event_type = "press",
                category   = cat,
                key_id     = ord("a") + (i % 26),
                iki        = iki,
            ))
            t += iki

        # ---- Run the window aggregator ----
        extractor = FeatureExtractor()
        sw = SlidingWindow(
            eye_queue       = eye_q,
            keystroke_queue = keystroke_q,
            window_size     = 7.0,
            step_size       = 2.0,
        )
        sw._drain_queue(eye_q,       sw._eye_buf)
        sw._drain_queue(keystroke_q, sw._key_buf)

        wf     = sw._compute_window(now - 7.0, now)
        result = extractor.predict(wf)

        # Verify face-signal features were captured
        fv = wf.feature_vector
        self.assertEqual(len(fv), 17)
        bfr = fv[7]
        mar = fv[8]
        asy = fv[9]
        self.assertLess(bfr, 0.45, f"BFR should be low (furrowed): {bfr:.3f}")
        self.assertLess(mar, 0.02, f"MAR should be tight (clenching): {mar:.3f}")
        self.assertGreater(asy, 0.01, f"Asymmetry should be elevated: {asy:.3f}")

        # Verify final classification
        self.assertIn(
            result.load_class,
            (CognitiveLoadClass.HIGH, CognitiveLoadClass.CRITICAL),
            f"Three-signal high-load scenario produced class "
            f"{result.load_class.value} (score={result.score:.1f})"
        )
        self.assertTrue(result.needs_intervention)


# ===========================================================================
# Run
# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
