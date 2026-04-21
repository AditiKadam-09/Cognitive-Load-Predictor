"""
ui/labeller.py
==============
Micro-labelling widget — pops up as a small "What's your load right now?"
prompt every N minutes.  User picks one of four load levels.
The (feature_vector, label) pair is appended to a CSV for later model training.

Updated for 17-feature three-signal model.
CSV columns: timestamp, label, [7 eye cols], [3 face cols], [7 keystroke cols]
"""

from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore  import Qt, QTimer, pyqtSignal
from PyQt6.QtGui   import QColor, QPainter, QPainterPath, QPen, QBrush
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QApplication,
                              QGraphicsDropShadowEffect)

from cognitive_monitor.analysis.feature_extractor import (
    CognitiveLoadClass,
)
from . import theme

logger = logging.getLogger(__name__)

DEFAULT_STORE = Path.home() / ".cognitive_monitor" / "training_data.csv"

# 17-feature CSV header: timestamp + label + 7 eye + 3 face + 7 keystroke
CSV_HEADER = ["timestamp", "label"] + [
    # Eye signal
    "mean_ear", "std_ear", "blink_rate", "gaze_stability",
    "perclos", "head_roll_mean", "face_coverage",
    # Face signal (new)
    "brow_furrow_ratio", "mouth_open_ratio", "eye_asymmetry",
    # Keystroke signal
    "typing_speed_cps", "mean_iki", "std_iki", "cv_iki",
    "backspace_rate", "burst_coefficient", "pause_count",
]


# ---------------------------------------------------------------------------
# Training data persistence
# ---------------------------------------------------------------------------
class TrainingDataStore:
    """Append-only CSV store for labelled 17-feature vectors."""

    def __init__(self, path: Path = DEFAULT_STORE) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADER)
        logger.info("TrainingDataStore at %s", path)

    def append(self, fv: np.ndarray, label: int) -> None:
        """
        Parameters
        ----------
        fv    : np.ndarray shape (17,)
        label : int  0=Low, 1=Medium, 2=High, 3=Critical
        """
        row = [time.time(), label] + fv.tolist()
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def load(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Returns (X, y) or None if fewer than 10 rows exist.
        X shape: (n, 17) — feature vectors
        y shape: (n,)    — integer labels
        """
        rows = []
        try:
            with open(self.path, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)   # skip header
                for row in reader:
                    if len(row) >= 19:   # 2 meta cols + 17 features
                        try:
                            rows.append([float(v) for v in row])
                        except ValueError:
                            continue
        except FileNotFoundError:
            return None

        if len(rows) < 10:
            return None

        arr = np.array(rows, dtype=np.float64)
        y   = arr[:, 1].astype(int)
        X   = arr[:, 2:].astype(np.float32)   # shape (n, 17)
        return X, y

    @property
    def sample_count(self) -> int:
        try:
            with open(self.path) as f:
                return max(0, sum(1 for _ in f) - 1)
        except FileNotFoundError:
            return 0


# ---------------------------------------------------------------------------
# Labeller pop-up
# ---------------------------------------------------------------------------
_LOAD_BUTTONS = [
    (CognitiveLoadClass.LOW,      0, "😌  Calm & Focused"),
    (CognitiveLoadClass.MEDIUM,   1, "🤔  Moderately Busy"),
    (CognitiveLoadClass.HIGH,     2, "😤  High Pressure"),
    (CognitiveLoadClass.CRITICAL, 3, "🤯  Overwhelmed"),
]


class LabellerPopup(QWidget):
    """Compact self-labelling widget. Emits `labelled(fv, int_label)` on pick."""

    labelled = pyqtSignal(object, int)   # (np.ndarray fv, int label)

    def __init__(
        self,
        feature_vector: Optional[np.ndarray] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._fv = feature_vector

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._build_ui()
        self._build_shadow()
        self._position()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 16)
        root.setSpacing(10)

        header = QLabel("How loaded do you feel right now?")
        header.setStyleSheet(f"""
            color: {theme.TEXT_PRIMARY};
            font-family: {theme.FONT_DISPLAY};
            font-size: 13px;
            font-weight: 700;
        """)
        root.addWidget(header)

        sub = QLabel("Your answer trains the AI model to recognise your patterns.")
        sub.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 10px;")
        sub.setWordWrap(True)
        root.addWidget(sub)

        for cls, label_int, text in _LOAD_BUTTONS:
            btn = QPushButton(text)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            accent = theme.accent_for(cls)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {theme.BG_RAISED};
                    color: {theme.TEXT_PRIMARY};
                    border: 1px solid {theme.BORDER};
                    border-radius: 8px;
                    padding: 7px 14px;
                    font-size: 12px;
                    text-align: left;
                    font-family: {theme.FONT_BODY};
                }}
                QPushButton:hover {{
                    background: {accent}22;
                    border-color: {accent};
                    color: {accent};
                }}
            """)
            btn.clicked.connect(lambda checked, li=label_int: self._pick(li))
            root.addWidget(btn)

        skip = QLabel("skip →")
        skip.setAlignment(Qt.AlignmentFlag.AlignRight)
        skip.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 10px;")
        skip.setCursor(Qt.CursorShape.PointingHandCursor)
        skip.mousePressEvent = lambda e: self.close()
        root.addWidget(skip)

        self.setFixedWidth(280)
        self.adjustSize()

    def _build_shadow(self) -> None:
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor("#00000088"))
        self.setGraphicsEffect(shadow)

    def _position(self) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 24,
                  screen.bottom() - self.height() - 24)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(1, 1, self.width() - 2, self.height() - 2, 14, 14)
        painter.fillPath(path, QBrush(QColor(theme.BG_SURFACE)))
        painter.setPen(QPen(QColor(theme.BORDER_GLOW), 1))
        painter.drawPath(path)

    def _pick(self, label_int: int) -> None:
        if self._fv is not None:
            self.labelled.emit(self._fv, label_int)
        self.close()


# ---------------------------------------------------------------------------
# LabellingScheduler
# ---------------------------------------------------------------------------
class LabellingScheduler:
    """
    Fires a LabellerPopup every `interval_minutes` minutes.
    The most-recent WindowedFeatures is passed to the popup so the
    label is aligned with the current feature state.
    """

    def __init__(
        self,
        store:            TrainingDataStore,
        interval_minutes: float = 10.0,
    ) -> None:
        self._store    = store
        self._interval = int(interval_minutes * 60 * 1000)
        self._last_fv: Optional[np.ndarray] = None
        self._timer    = QTimer()
        self._timer.setInterval(self._interval)
        self._timer.timeout.connect(self._prompt)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def update_features(self, wf) -> None:
        self._last_fv = wf.feature_vector.copy()

    def _prompt(self) -> None:
        popup = LabellerPopup(feature_vector=self._last_fv)
        popup.labelled.connect(self._on_labelled)
        popup.show()

    def _on_labelled(self, fv: np.ndarray, label: int) -> None:
        self._store.append(fv, label)
        logger.info("Label saved: class=%d  total=%d",
                    label, self._store.sample_count)
