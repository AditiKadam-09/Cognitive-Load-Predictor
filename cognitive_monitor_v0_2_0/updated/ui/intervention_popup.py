"""
ui/intervention_popup.py
=========================
Aesthetically-pleasing intervention pop-up that slides in from the
bottom-right corner of the screen.

Design: "biometric terminal" aesthetic.
  • Frameless window with custom-painted rounded corners
  • State-reactive accent stripe (HIGH = orange glow, CRITICAL = red pulse)
  • Animated slide-in / fade-in on show; fade-out on dismiss
  • Auto-dismiss after `auto_close_ms` milliseconds
  • Dismissible by button click OR by clicking anywhere on the card

Pop-up is emitted on the Qt main thread via Qt signals so it is safe
to call `show_intervention()` from any background thread.
"""

from __future__ import annotations

import time
from typing import Optional

from PyQt6.QtCore  import (Qt, QTimer, QPropertyAnimation,
                            QEasingCurve, pyqtProperty, QPoint,
                            QSequentialAnimationGroup, QRect)
from PyQt6.QtGui   import (QColor, QPainter, QPainterPath, QFont,
                            QLinearGradient, QRadialGradient,
                            QBrush, QPen, QScreen)
from PyQt6.QtWidgets import (QWidget, QLabel, QPushButton,
                              QVBoxLayout, QHBoxLayout,
                              QApplication, QGraphicsDropShadowEffect,
                              QSizePolicy)

from cognitive_monitor.analysis.feature_extractor import CognitiveLoadClass
from . import theme


# ---------------------------------------------------------------------------
# Helper: action items per load class
# ---------------------------------------------------------------------------
_ACTIONS = {
    CognitiveLoadClass.HIGH: [
        ("🫁", "Box breathing — 4s in, 4s hold, 4s out, 4s hold"),
        ("👁", "Look at something 20 m away for 20 seconds"),
        ("💧", "Drink a glass of water"),
    ],
    CognitiveLoadClass.CRITICAL: [
        ("🚶", "Step away from the screen for 5 minutes"),
        ("💧", "Hydrate — drink water now"),
        ("🤸", "Stand up and stretch your neck and shoulders"),
    ],
}

_HEADLINES = {
    CognitiveLoadClass.HIGH:     "Elevated Cognitive Load",
    CognitiveLoadClass.CRITICAL: "Critical Load Detected",
}

_SUBTITLES = {
    CognitiveLoadClass.HIGH:     "Your focus signals are peaking. A short reset will restore performance.",
    CognitiveLoadClass.CRITICAL: "Sustained overload detected. Rest now to protect your wellbeing.",
}

POPUP_W = 380
POPUP_H = 240   # dynamic — grows with action items
CORNER_R = 16


class InterventionPopup(QWidget):
    """
    Frameless, always-on-top intervention card.

    Parameters
    ----------
    load_class    : CognitiveLoadClass  (HIGH or CRITICAL)
    score         : float               cognitive load score 0-100
    auto_close_ms : int                 ms before auto-dismiss (0 = never)
    parent        : optional QWidget
    """

    def __init__(
        self,
        load_class:    CognitiveLoadClass = CognitiveLoadClass.HIGH,
        score:         float = 70.0,
        auto_close_ms: int   = 30_000,   # 30 seconds
        parent:        Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.load_class    = load_class
        self.score         = score
        self._auto_close   = auto_close_ms
        self._accent_color = QColor(theme.accent_for(load_class))
        self._opacity      = 0.0

        self._build_window()
        self._build_layout()
        self._build_shadow()
        self._build_animations()

        if auto_close_ms > 0:
            QTimer.singleShot(auto_close_ms, self._dismiss)

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------
    def _build_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool             # no taskbar entry
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedWidth(POPUP_W)

    def _build_layout(self) -> None:
        accent   = theme.accent_for(self.load_class)
        headline = _HEADLINES.get(self.load_class, "Cognitive Load Alert")
        subtitle = _SUBTITLES.get(self.load_class, "")
        actions  = _ACTIONS.get(self.load_class, [])

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 18)
        root.setSpacing(0)

        # ---- Top row: icon badge + headline + score ----
        top = QHBoxLayout()
        top.setSpacing(12)

        badge = QLabel("⚡" if self.load_class == CognitiveLoadClass.CRITICAL else "⚠")
        badge.setStyleSheet(f"""
            color: {accent};
            font-size: 22px;
            padding: 0px;
        """)
        badge.setFixedSize(32, 32)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        hl = QLabel(headline)
        hl.setStyleSheet(f"""
            color: {theme.TEXT_PRIMARY};
            font-family: {theme.FONT_DISPLAY};
            font-size: 15px;
            font-weight: 700;
            letter-spacing: 0.3px;
        """)
        sub = QLabel(subtitle)
        sub.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 11px;")
        sub.setWordWrap(True)
        title_col.addWidget(hl)
        title_col.addWidget(sub)

        score_badge = QLabel(f"{self.score:.0f}")
        score_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_badge.setFixedSize(44, 44)
        score_badge.setStyleSheet(f"""
            color: {accent};
            background: transparent;
            border: 2px solid {accent};
            border-radius: 22px;
            font-family: {theme.FONT_MONO};
            font-size: 15px;
            font-weight: 700;
        """)

        top.addWidget(badge)
        top.addLayout(title_col, 1)
        top.addWidget(score_badge)
        root.addLayout(top)

        root.addSpacing(14)

        # ---- Divider ----
        divider = QWidget()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background: {theme.BORDER};")
        root.addWidget(divider)
        root.addSpacing(12)

        # ---- Action items ----
        for icon, text in actions:
            row = QHBoxLayout()
            row.setSpacing(10)
            ic = QLabel(icon)
            ic.setStyleSheet("font-size: 15px;")
            ic.setFixedWidth(22)
            tx = QLabel(text)
            tx.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: 12px;")
            tx.setWordWrap(True)
            row.addWidget(ic)
            row.addWidget(tx, 1)
            root.addLayout(row)
            root.addSpacing(6)

        root.addSpacing(10)

        # ---- Bottom row: timer label + dismiss button ----
        bottom = QHBoxLayout()
        self._timer_label = QLabel("")
        self._timer_label.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px; font-family: {theme.FONT_MONO};")

        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.setFixedHeight(32)
        dismiss_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        dismiss_btn.setStyleSheet(f"""
            QPushButton {{
                background: {theme.BG_RAISED};
                color: {accent};
                border: 1px solid {accent}55;
                border-radius: 8px;
                padding: 0 20px;
                font-size: 12px;
                font-weight: 600;
                font-family: {theme.FONT_BODY};
            }}
            QPushButton:hover {{
                background: {accent}22;
                border-color: {accent};
            }}
            QPushButton:pressed {{
                background: {accent}44;
            }}
        """)
        dismiss_btn.clicked.connect(self._dismiss)

        bottom.addWidget(self._timer_label, 1)
        bottom.addWidget(dismiss_btn)
        root.addLayout(bottom)

        self.adjustSize()

        # Timer countdown
        if self._auto_close > 0:
            self._countdown = self._auto_close // 1000
            self._update_timer_label()
            tick = QTimer(self)
            tick.setInterval(1000)
            tick.timeout.connect(self._tick_countdown)
            tick.start()

    def _build_shadow(self) -> None:
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(theme.glow_for(self.load_class) + "88"))
        self.setGraphicsEffect(shadow)

    def _build_animations(self) -> None:
        # Opacity animation (fade in / out)
        self._opacity_anim = QPropertyAnimation(self, b"windowOpacity")
        self._opacity_anim.setDuration(280)
        self._opacity_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    # ------------------------------------------------------------------
    # Custom painting — rounded card
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect  = self.rect().adjusted(2, 2, -2, -2)
        path  = QPainterPath()
        path.addRoundedRect(rect.x(), rect.y(),
                            rect.width(), rect.height(),
                            CORNER_R, CORNER_R)

        # Background gradient
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0, QColor(theme.BG_SURFACE))
        grad.setColorAt(1, QColor(theme.BG_DEEP))
        painter.fillPath(path, QBrush(grad))

        # Accent top stripe
        stripe_path = QPainterPath()
        stripe_path.addRoundedRect(rect.x(), rect.y(),
                                   rect.width(), 3,
                                   CORNER_R, CORNER_R)
        # Clip to top half only
        clip = QPainterPath()
        clip.addRect(rect.x(), rect.y(), rect.width(), 8)
        stripe_path = stripe_path.intersected(clip)
        painter.fillPath(stripe_path, QBrush(self._accent_color))

        # Border
        border_color = QColor(theme.accent_for(self.load_class) + "55")
        painter.setPen(QPen(border_color, 1))
        painter.drawPath(path)

    # ------------------------------------------------------------------
    # Show / dismiss helpers
    # ------------------------------------------------------------------
    def show_at_bottom_right(self) -> None:
        """Position at bottom-right of primary screen and animate in."""
        screen: QScreen = QApplication.primaryScreen()
        geom   = screen.availableGeometry()
        self.adjustSize()
        x = geom.right()  - self.width()  - 24
        y = geom.bottom() - self.height() - 24
        self.move(x, y)

        self.setWindowOpacity(0.0)
        self.show()
        self._opacity_anim.setStartValue(0.0)
        self._opacity_anim.setEndValue(1.0)
        self._opacity_anim.start()

    def _dismiss(self) -> None:
        """Animate out then close. Guard against double-call."""
        if getattr(self, '_dismissing', False):
            return
        self._dismissing = True
        self._opacity_anim.stop()
        self._opacity_anim.setStartValue(self.windowOpacity())
        self._opacity_anim.setEndValue(0.0)
        self._opacity_anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self._opacity_anim.finished.connect(self.close)
        self._opacity_anim.start()

    def _tick_countdown(self) -> None:
        self._countdown -= 1
        self._update_timer_label()

    def _update_timer_label(self) -> None:
        if hasattr(self, "_timer_label"):
            self._timer_label.setText(f"auto-dismiss in {self._countdown}s")

    # Allow clicking the card body to dismiss too
    def mousePressEvent(self, event) -> None:
        self._dismiss()
