"""
analysis/feature_extractor.py
==============================
Multimodal fusion layer and Cognitive Load classifier.

Three-signal pipeline
---------------------
WindowedFeatures (17 features)
  → FeatureExtractor.transform()
  → StandardScaler (fitted at runtime)
  → RandomForestClassifier  (4 classes)
  → CognitiveLoadResult (score 0-100, class label)

Signal blocks in the feature vector
-------------------------------------
  [0–6]   Eye signal    : EAR, blink, gaze, PERCLOS, head roll, face coverage
  [7–9]   Face signal   : brow furrow ratio, mouth open ratio, eye asymmetry
  [10–16] Keystroke sig : speed, IKI stats, backspace, burst, pause

Heuristic cold-start scorer (pre-ML)
--------------------------------------
Before enough labelled data exists the system uses a weighted rule-based
scorer.  The face-signal block contributes three new components:

  brow_furrow  – low BFR (furrowed brows) → high load
  mouth_tension – extreme MAR (tight or open mouth) → elevated load
  eye_asymmetry – high |L-R EAR| → facial tension → elevated load

Weights are re-calibrated from v0 to sum to 1.0 across 9 components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cognitive load levels
# ---------------------------------------------------------------------------
class CognitiveLoadClass(Enum):
    LOW      = "Low"
    MEDIUM   = "Medium"
    HIGH     = "High"
    CRITICAL = "Critical"


@dataclass
class CognitiveLoadResult:
    """Output of one prediction cycle."""
    timestamp:       float
    score:           float
    load_class:      CognitiveLoadClass
    confidence:      float
    feature_vector:  np.ndarray   # 17-D vector
    source:          str          # "heuristic" | "ml_model"

    @property
    def needs_intervention(self) -> bool:
        return self.load_class in (CognitiveLoadClass.HIGH, CognitiveLoadClass.CRITICAL)

    @property
    def intervention_message(self) -> Optional[str]:
        if self.load_class == CognitiveLoadClass.HIGH:
            return (
                "Your cognitive load is elevated.\n"
                "Try a 2-minute box-breathing exercise:\n"
                "Inhale 4s → Hold 4s → Exhale 4s → Hold 4s."
            )
        if self.load_class == CognitiveLoadClass.CRITICAL:
            return (
                "Critical load detected.\n"
                "Step away from the screen for 5 minutes.\n"
                "Hydrate, stretch, and rest your eyes."
            )
        return None


# ---------------------------------------------------------------------------
# Score thresholds (0–100)
# ---------------------------------------------------------------------------
_THRESHOLDS = {
    CognitiveLoadClass.LOW:      (0,  30),
    CognitiveLoadClass.MEDIUM:   (30, 60),
    CognitiveLoadClass.HIGH:     (60, 80),
    CognitiveLoadClass.CRITICAL: (80, 100),
}


def score_to_class(score: float) -> CognitiveLoadClass:
    for cls, (lo, hi) in _THRESHOLDS.items():
        if lo <= score < hi:
            return cls
    return CognitiveLoadClass.CRITICAL


# ---------------------------------------------------------------------------
# Heuristic scorer (rule-based fallback / cold-start)
# ---------------------------------------------------------------------------
class HeuristicScorer:
    """
    Maps a 17-D WindowedFeatures vector to a 0–100 score using
    handcrafted thresholds across all three signal blocks.

    Scoring components (each 0–1, weighted sum scaled to 0–100):

    Eye signal (total weight 0.48)
      • ear           0.18  — low EAR → higher load
      • perclos       0.14  — high fraction closed → higher load
      • gaze          0.10  — erratic gaze → higher load
      • blink_anomaly 0.06  — deviate from 15 blinks/min → higher load

    Face signal — NEW (total weight 0.22)
      • brow_furrow   0.10  — low BFR → furrowed brows → higher load
      • mouth_tension 0.07  — extreme MAR (≈0 or >0.15) → higher load
      • eye_asymmetry 0.05  — high |L-R EAR| → facial tension

    Keystroke signal (total weight 0.30)
      • typing        0.15  — high cv_iki + burst → higher load
      • backspace     0.15  — high error rate → higher load

    Total: 0.18+0.14+0.10+0.06 + 0.10+0.07+0.05 + 0.15+0.15 = 1.00
    """

    # Eye / blink reference values
    NORMAL_EAR        = 0.30
    NORMAL_BLINK_RATE = 15.0   # blinks/min
    BLINK_RATE_LOW    = 8.0
    BLINK_RATE_HIGH   = 30.0

    # Face-signal reference values  (NEW)
    NORMAL_BFR        = 0.55   # relaxed brow-furrow ratio
    MIN_BFR_STRESS    = 0.35   # at or below → furrowed
    NORMAL_MAR        = 0.03   # resting mouth open ratio
    STRESS_MAR_TIGHT  = 0.01   # jaw clenching
    STRESS_MAR_OPEN   = 0.15   # wide open (fatigue/gasp)
    NORMAL_ASYMMETRY  = 0.02   # small asymmetry is normal
    HIGH_ASYMMETRY    = 0.08   # clearly elevated

    # Keystroke reference values
    NORMAL_BACKSPACE  = 0.05

    # Component weights — must sum to 1.0
    WEIGHTS = {
        # Eye
        "ear":           0.18,
        "perclos":       0.14,
        "gaze":          0.10,
        "blink_anomaly": 0.06,
        # Face (new)
        "brow_furrow":   0.10,
        "mouth_tension": 0.07,
        "eye_asymmetry": 0.05,
        # Keystroke
        "typing":        0.15,
        "backspace":     0.15,
    }

    def score(self, fv: np.ndarray) -> float:
        """
        Parameters
        ----------
        fv : np.ndarray shape (17,)

        Returns
        -------
        float in [0, 100]
        """
        (mean_ear, std_ear, blink_rate, gaze_stability,
         perclos, head_roll_mean, face_coverage,
         brow_furrow_ratio, mouth_open_ratio, eye_asymmetry,
         typing_speed_cps, mean_iki, std_iki, cv_iki,
         backspace_rate, burst_coefficient, pause_count) = fv

        # ---- Eye: low EAR → high load ----
        ear_drop  = max(0.0, self.NORMAL_EAR - mean_ear) / self.NORMAL_EAR
        ear_score = float(np.clip(ear_drop * 2.0, 0, 1))

        # ---- PERCLOS ----
        perclos_score = float(np.clip(perclos * 3.0, 0, 1))

        # ---- Gaze instability ----
        gaze_score = float(np.clip(1.0 - gaze_stability, 0, 1))

        # ---- Blink anomaly ----
        blink_dev   = abs(blink_rate - self.NORMAL_BLINK_RATE) / self.NORMAL_BLINK_RATE
        blink_score = float(np.clip(blink_dev, 0, 1))

        # ---- NEW: Brow furrow — low ratio = furrowed = stressed ----
        # Score rises as BFR drops below NORMAL_BFR toward MIN_BFR_STRESS
        bfr_drop       = max(0.0, self.NORMAL_BFR - brow_furrow_ratio)
        bfr_range      = max(self.NORMAL_BFR - self.MIN_BFR_STRESS, 1e-6)
        brow_score     = float(np.clip(bfr_drop / bfr_range, 0, 1))

        # ---- NEW: Mouth tension — extreme MAR (too tight OR too open) ----
        # Two stress zones: jaw clenching (very tight) and wide open (fatigue)
        tight_score = float(np.clip(
            (self.STRESS_MAR_TIGHT - mouth_open_ratio) / max(self.STRESS_MAR_TIGHT, 1e-6),
            0, 1,
        ))
        open_score = float(np.clip(
            (mouth_open_ratio - self.NORMAL_MAR) / (self.STRESS_MAR_OPEN - self.NORMAL_MAR + 1e-6),
            0, 1,
        ))
        mouth_score = float(np.clip(max(tight_score, open_score), 0, 1))

        # ---- NEW: Eye asymmetry ----
        asym_range  = max(self.HIGH_ASYMMETRY - self.NORMAL_ASYMMETRY, 1e-6)
        asym_score  = float(np.clip(
            (eye_asymmetry - self.NORMAL_ASYMMETRY) / asym_range, 0, 1
        ))

        # ---- Keystroke irregularity: high CV_IKI + high burst ----
        typing_score = float(np.clip((cv_iki + burst_coefficient) / 2.0, 0, 1))

        # ---- Backspace rate ----
        bs_score = float(np.clip(
            (backspace_rate - self.NORMAL_BACKSPACE) / (1.0 - self.NORMAL_BACKSPACE),
            0, 1,
        ))

        components = {
            "ear":           ear_score,
            "perclos":       perclos_score,
            "gaze":          gaze_score,
            "blink_anomaly": blink_score,
            "brow_furrow":   brow_score,
            "mouth_tension": mouth_score,
            "eye_asymmetry": asym_score,
            "typing":        typing_score,
            "backspace":     bs_score,
        }

        weighted  = sum(components[k] * w for k, w in self.WEIGHTS.items())
        raw_score = float(np.clip(weighted * 100.0 * 1.2, 0, 100))
        return raw_score


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """
    Facade that accepts a `WindowedFeatures` object and returns a
    `CognitiveLoadResult`.

    Starts with `HeuristicScorer`.  After `fit()` is called with labelled
    data, switches to the trained ML model.
    """

    def __init__(self) -> None:
        self._heuristic = HeuristicScorer()
        self._scaler    = None
        self._model     = None
        self._use_model = False

    def predict(self, wf) -> CognitiveLoadResult:
        fv = wf.feature_vector   # shape (17,)

        if self._use_model and self._model is not None:
            return self._ml_predict(fv, wf.timestamp)
        return self._heuristic_predict(fv, wf.timestamp)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the RandomForest fusion model on 17-feature vectors.

        Parameters
        ----------
        X : np.ndarray shape (n_samples, 17)
        y : np.ndarray shape (n_samples,)  — integer labels 0-3
        """
        try:
            from sklearn.ensemble      import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline      import Pipeline

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )),
            ])
            pipeline.fit(X, y)
            self._model     = pipeline
            self._use_model = True
            logger.info("ML model trained on %d samples (17 features).", len(X))
        except ImportError:
            logger.error("scikit-learn not available; staying with heuristic scorer.")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    def _heuristic_predict(self, fv: np.ndarray, ts: float) -> CognitiveLoadResult:
        score = self._heuristic.score(fv)
        cls   = score_to_class(score)
        return CognitiveLoadResult(
            timestamp=ts,
            score=score,
            load_class=cls,
            confidence=0.6,
            feature_vector=fv,
            source="heuristic",
        )

    def _ml_predict(self, fv: np.ndarray, ts: float) -> CognitiveLoadResult:
        try:
            proba_raw = self._model.predict_proba(fv.reshape(1, -1))[0]
        except Exception:
            logger.warning("ML predict failed; falling back to heuristic.")
            return self._heuristic_predict(fv, ts)

        n_classes = len(proba_raw)
        if n_classes < 4:
            proba = np.zeros(4, dtype=np.float64)
            proba[:n_classes] = proba_raw
            proba /= proba.sum() + 1e-9
        else:
            proba = proba_raw

        label = int(np.argmax(proba))
        conf  = float(proba[label])
        cls   = list(CognitiveLoadClass)[label]

        midpoints = [15.0, 45.0, 70.0, 90.0]
        score     = float(np.dot(proba, midpoints))

        if conf < 0.4:
            h_result = self._heuristic_predict(fv, ts)
            score    = 0.5 * score + 0.5 * h_result.score
            cls      = score_to_class(score)

        return CognitiveLoadResult(
            timestamp=ts,
            score=score,
            load_class=cls,
            confidence=conf,
            feature_vector=fv,
            source="ml_model",
        )
