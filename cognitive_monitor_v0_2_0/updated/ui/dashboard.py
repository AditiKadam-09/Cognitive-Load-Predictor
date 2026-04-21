"""
ui/dashboard.py
===============
Live dashboard window.

Updated for 17-feature three-signal model:
  • Feature bars now include brow_furrow_ratio, mouth_open_ratio,
    eye_asymmetry  (the three new face-signal features at indices 7–9)
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional

from PyQt6.QtCore  import (Qt, QTimer, pyqtSignal, pyqtSlot,
                            QRectF, QPointF, QSize)
from PyQt6.QtGui   import (QPainter, QPainterPath, QColor, QFont,
                            QPen, QBrush, QLinearGradient,
                            QConicalGradient, QFontMetrics, QPolygonF,
                            QRadialGradient)
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QFrame, QSizePolicy,
                              QGraphicsDropShadowEffect)

from cognitive_monitor.analysis.feature_extractor import (
    CognitiveLoadResult, CognitiveLoadClass
)
from . import theme

HISTORY_LEN = 60   # last 60 windows (~120 s at step=2 s)


# ---------------------------------------------------------------------------
# Radial gauge
# ---------------------------------------------------------------------------
class RadialGauge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._score      = 0.0
        self._cls        = CognitiveLoadClass.LOW
        self._anim_score = 0.0
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._animate_tick)

    def set_score(self, score: float, cls: CognitiveLoadClass) -> None:
        self._score = score
        self._cls   = cls
        if not self._timer.isActive():
            self._timer.start()

    def _animate_tick(self) -> None:
        diff = self._score - self._anim_score
        if abs(diff) < 0.5:
            self._anim_score = self._score
            self._timer.stop()
        else:
            self._anim_score += diff * 0.12
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        side = min(self.width(), self.height()) - 20
        cx, cy = self.width() / 2, self.height() / 2
        r      = side / 2
        thick  = max(12, side * 0.10)
        rect   = QRectF(cx - r, cy - r, side, side)

        pen_track = QPen(QColor(theme.BORDER), thick, Qt.PenStyle.SolidLine,
                         Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_track)
        painter.drawArc(rect, int((-135 + 180) * 16), int(-270 * 16))

        frac    = self._anim_score / 100.0
        accent  = QColor(theme.accent_for(self._cls))
        pen_val = QPen(accent, thick, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_val)
        painter.drawArc(rect, int((-135 + 180) * 16), int(-270 * frac * 16))

        painter.setPen(QPen(QColor(theme.TEXT_PRIMARY)))
        font = QFont("JetBrains Mono, Consolas, monospace")
        font.setPixelSize(int(side * 0.22))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect.adjusted(0, side * 0.05, 0, 0),
                         Qt.AlignmentFlag.AlignCenter,
                         f"{self._anim_score:.0f}")

        font2 = QFont(theme.FONT_BODY)
        font2.setPixelSize(int(side * 0.09))
        painter.setFont(font2)
        painter.setPen(QPen(QColor(theme.accent_for(self._cls))))
        painter.drawText(rect.adjusted(0, side * 0.28, 0, 0),
                         Qt.AlignmentFlag.AlignCenter,
                         theme.label_for(self._cls))


# ---------------------------------------------------------------------------
# Feature bar
# ---------------------------------------------------------------------------
class FeatureBar(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label  = label
        self._value  = 0.0
        self._accent = theme.ACCENT_DEFAULT
        self.setFixedHeight(22)
        self.setMinimumWidth(100)

    def set_value(self, value: float, accent: str = theme.ACCENT_DEFAULT) -> None:
        self._value  = max(0.0, min(1.0, value))
        self._accent = accent
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        label_w = 140
        bar_x   = label_w + 8
        bar_w   = w - bar_x - 8
        bar_h   = 6
        bar_y   = (h - bar_h) / 2

        font = QFont(theme.FONT_MONO)
        font.setPixelSize(10)
        painter.setFont(font)
        painter.setPen(QPen(QColor(theme.TEXT_SECONDARY)))
        painter.drawText(0, 0, label_w, h,
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         self._label)

        painter.setBrush(QBrush(QColor(theme.BG_RAISED)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 3, 3)

        fill_w = bar_w * self._value
        if fill_w > 0:
            painter.setBrush(QBrush(QColor(self._accent)))
            painter.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 3, 3)

        pct_str = f"{self._value * 100:.0f}"
        font2   = QFont(theme.FONT_MONO)
        font2.setPixelSize(10)
        painter.setFont(font2)
        painter.setPen(QPen(QColor(theme.TEXT_DIM)))
        painter.drawText(int(bar_x + fill_w + 4), 0, 40, h,
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         pct_str)


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------
class Sparkline(QWidget):
    def __init__(self, history_len: int = HISTORY_LEN, parent=None):
        super().__init__(parent)
        self._data: deque[float] = deque([0.0] * history_len, maxlen=history_len)
        self._cls                = CognitiveLoadClass.LOW
        self.setMinimumHeight(50)

    def append(self, score: float, cls: CognitiveLoadClass) -> None:
        self._data.append(score)
        self._cls = cls
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        pad  = 4
        n    = len(self._data)
        accent = QColor(theme.accent_for(self._cls))
        if n < 2:
            return

        def px(i): return pad + (i / (n - 1)) * (w - 2 * pad)
        def py(v): return (h - pad) - (v / 100.0) * (h - 2 * pad)

        path = QPainterPath()
        data = list(self._data)
        path.moveTo(px(0), py(data[0]))
        for i, v in enumerate(data):
            path.lineTo(px(i), py(v))
        path.lineTo(px(n - 1), h)
        path.lineTo(px(0), h)
        path.closeSubpath()

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor(theme.accent_for(self._cls) + "55"))
        grad.setColorAt(1, QColor(theme.accent_for(self._cls) + "00"))
        painter.fillPath(path, QBrush(grad))

        line_path = QPainterPath()
        line_path.moveTo(px(0), py(data[0]))
        for i, v in enumerate(data):
            line_path.lineTo(px(i), py(v))

        painter.setPen(QPen(accent, 1.5, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(line_path)

        font = QFont(theme.FONT_MONO)
        font.setPixelSize(9)
        painter.setFont(font)
        painter.setPen(QPen(QColor(theme.TEXT_DIM)))
        painter.drawText(0, 0, 20, h,
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "100")
        painter.drawText(0, h // 2 - 6, 20, 12,
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "50")
        painter.drawText(0, h - 12, 20, 12,
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "0")


def _section(title: str) -> QLabel:
    lbl = QLabel(title.upper())
    lbl.setStyleSheet(f"""
        color: {theme.TEXT_DIM};
        font-family: {theme.FONT_MONO};
        font-size: 10px;
        letter-spacing: 1.5px;
        padding-bottom: 4px;
        border-bottom: 1px solid {theme.BORDER};
    """)
    return lbl


# ---------------------------------------------------------------------------
# DashboardWindow
# ---------------------------------------------------------------------------
class DashboardWindow(QWidget):
    _result_received = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CognitiveMonitor — Dashboard")
        self.setMinimumSize(560, 680)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setStyleSheet(theme.APP_STYLESHEET)
        self._build_ui()
        self._result_received.connect(self._apply_result)

    def update_result(self, result: CognitiveLoadResult) -> None:
        self._result_received.emit(result)

    def closeEvent(self, event) -> None:
        event.ignore()
        self.hide()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # Title bar
        title_row = QHBoxLayout()
        app_title = QLabel("COGNITIVE MONITOR")
        app_title.setStyleSheet(f"""
            font-family: {theme.FONT_DISPLAY};
            font-size: 16px;
            font-weight: 800;
            letter-spacing: 2px;
            color: {theme.TEXT_PRIMARY};
        """)
        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet(f"color: {theme.ACCENT_LOW}; font-size: 16px;")
        self._source_lbl = QLabel("heuristic")
        self._source_lbl.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 10px; font-family: {theme.FONT_MONO};")
        title_row.addWidget(app_title, 1)
        title_row.addWidget(self._source_lbl)
        title_row.addSpacing(8)
        title_row.addWidget(self._status_dot)
        root.addLayout(title_row)

        # Gauge + metadata
        gauge_row = QHBoxLayout()
        gauge_row.setSpacing(20)
        self._gauge = RadialGauge()
        self._gauge.setFixedSize(180, 180)
        gauge_row.addWidget(self._gauge)

        meta_col = QVBoxLayout()
        meta_col.setSpacing(8)
        meta_col.setAlignment(Qt.AlignmentFlag.AlignTop)

        def _meta_row(label: str) -> tuple[QLabel, QLabel]:
            row = QHBoxLayout()
            row.setSpacing(8)
            k = QLabel(label)
            k.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px; font-family: {theme.FONT_MONO};")
            k.setFixedWidth(110)
            v = QLabel("—")
            v.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: 12px; font-family: {theme.FONT_MONO};")
            row.addWidget(k)
            row.addWidget(v, 1)
            meta_col.addLayout(row)
            return k, v

        _, self._mv_score     = _meta_row("SCORE")
        _, self._mv_class     = _meta_row("CLASS")
        _, self._mv_conf      = _meta_row("CONFIDENCE")
        _, self._mv_ear       = _meta_row("EAR")
        _, self._mv_blink     = _meta_row("BLINK RATE")
        _, self._mv_brow      = _meta_row("BROW FURROW")   # NEW
        _, self._mv_mouth     = _meta_row("MOUTH OPEN")    # NEW
        _, self._mv_asym      = _meta_row("EYE ASYM")      # NEW
        _, self._mv_typing    = _meta_row("TYPING SPD")
        _, self._mv_backspace = _meta_row("BACKSPACE %")
        _, self._mv_ts        = _meta_row("UPDATED")

        gauge_row.addLayout(meta_col, 1)
        root.addLayout(gauge_row)

        # Feature bars — 17 features
        root.addWidget(_section("Feature Vector (Eye · Face · Keystroke)"))
        self._feature_names = [
            # Eye signal (7)
            "mean_ear", "std_ear", "blink_rate", "gaze_stability",
            "perclos", "head_roll", "face_coverage",
            # Face signal (3) — NEW
            "brow_furrow_ratio", "mouth_open_ratio", "eye_asymmetry",
            # Keystroke signal (7)
            "typing_speed", "mean_iki", "std_iki",
            "cv_iki", "backspace_rate", "burst_coeff", "pause_count",
        ]
        # Max values for normalising each bar to [0,1]
        self._feature_max = [
            # Eye
            0.4,   # mean_ear
            0.1,   # std_ear
            40.0,  # blink_rate
            1.0,   # gaze_stability
            1.0,   # perclos
            15.0,  # head_roll
            1.0,   # face_coverage
            # Face (new)
            1.0,   # brow_furrow_ratio  (already 0–1)
            0.30,  # mouth_open_ratio   (>0.30 = wide open)
            0.15,  # eye_asymmetry      (>0.15 = very asymmetric)
            # Keystroke
            10.0,  # typing_speed
            0.5,   # mean_iki
            0.3,   # std_iki
            3.0,   # cv_iki
            1.0,   # backspace_rate
            1.0,   # burst_coeff
            10.0,  # pause_count
        ]
        self._bars: list[FeatureBar] = []
        bar_grid = QVBoxLayout()
        bar_grid.setSpacing(3)
        for name in self._feature_names:
            bar = FeatureBar(name)
            self._bars.append(bar)
            bar_grid.addWidget(bar)
        root.addLayout(bar_grid)

        # Sparkline
        root.addWidget(_section("Score History (last 2 min)"))
        self._sparkline = Sparkline()
        self._sparkline.setMinimumHeight(60)
        root.addWidget(self._sparkline)

    @pyqtSlot(object)
    def _apply_result(self, result: CognitiveLoadResult) -> None:
        cls    = result.load_class
        accent = theme.accent_for(cls)

        self._gauge.set_score(result.score, cls)
        self._sparkline.append(result.score, cls)

        self._status_dot.setStyleSheet(f"color: {accent}; font-size: 16px;")
        self._source_lbl.setText(result.source)

        self._mv_score.setText(f"{result.score:.1f}")
        self._mv_class.setText(theme.label_for(cls))
        self._mv_class.setStyleSheet(f"color: {accent}; font-size: 12px; font-family: {theme.FONT_MONO};")
        self._mv_conf.setText(f"{result.confidence * 100:.0f}%")
        self._mv_ts.setText(time.strftime("%H:%M:%S", time.localtime(result.timestamp)))

        fv = result.feature_vector
        for i, bar in enumerate(self._bars):
            if i < len(fv):
                raw  = float(fv[i])
                mx   = self._feature_max[i]
                norm = min(raw / max(mx, 1e-6), 1.0)
                bar.set_value(norm, accent)

        # Populate metadata from feature vector (17-element layout)
        if len(fv) >= 17:
            self._mv_ear.setText(f"{fv[0]:.3f}")
            self._mv_blink.setText(f"{fv[2]:.1f}/min")
            self._mv_brow.setText(f"{fv[7]:.3f}")   # brow_furrow_ratio
            self._mv_mouth.setText(f"{fv[8]:.3f}")   # mouth_open_ratio
            self._mv_asym.setText(f"{fv[9]:.3f}")    # eye_asymmetry
            self._mv_typing.setText(f"{fv[10]:.1f} cps")
            self._mv_backspace.setText(f"{fv[14] * 100:.1f}%")
