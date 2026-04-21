"""
ui/tray_app.py
==============
System-tray application — the top-level PyQt6 controller.

Responsibilities
----------------
• Owns the QApplication event loop
• Hosts a QSystemTrayIcon with a context menu
• Receives CognitiveLoadResult objects via a Qt signal from the
  acquisition pipeline (safe for cross-thread calls)
• Triggers InterventionPopup for HIGH / CRITICAL states (with cooldown)
• Opens / hides the DashboardWindow
• Schedules the LabellerPopup and routes labelled data to FeatureExtractor
  for online retraining

Tray icon
---------
The icon is a tiny programmatically-drawn circle whose fill colour
matches the current load class:
  Low      → teal  #2DD4BF
  Medium   → amber #F5A623
  High     → orange #F97316
  Critical → pulsing red #EF4444

FIXES APPLIED
-------------
1. CognitiveMonitorApp now inherits QObject so PyQt6 signal/slot
   connections work correctly (was plain Python class → TypeError on connect).
2. super().__init__() called as QObject.__init__ in __init__.
3. QAction import moved from QtGui to QtGui (already correct in PyQt6,
   but added explicit QObject import from QtCore).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from PyQt6.QtCore  import (Qt, QTimer, pyqtSignal, pyqtSlot, QThread,
                            QSize, QObject)
from PyQt6.QtGui   import (QIcon, QPixmap, QPainter, QColor,
                            QBrush, QRadialGradient, QAction)
from PyQt6.QtWidgets import (QApplication, QSystemTrayIcon, QMenu,
                              QWidget)

from cognitive_monitor.analysis.feature_extractor import (
    CognitiveLoadResult, CognitiveLoadClass, FeatureExtractor
)
from . import theme
from .dashboard          import DashboardWindow
from .intervention_popup import InterventionPopup
from .labeller           import TrainingDataStore, LabellingScheduler

logger = logging.getLogger(__name__)

INTERVENTION_COOLDOWN_S = 120.0   # 2 minutes between pop-ups
TRAY_ICON_SIZE          = 22      # pixels


# ---------------------------------------------------------------------------
# Tray icon generator
# ---------------------------------------------------------------------------
def _make_tray_icon(cls: CognitiveLoadClass, pulse: bool = False) -> QIcon:
    """
    Render a small coloured circle as the tray icon.
    When `pulse=True` the icon has a lighter outer ring (critical state).
    """
    size   = TRAY_ICON_SIZE * 2   # render at 2× for HiDPI
    px     = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    accent = QColor(theme.accent_for(cls))

    painter = QPainter(px)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    cx = cy = size / 2
    r  = size / 2 - 3

    if pulse:
        # Outer glow ring
        outer = QColor(accent)
        outer.setAlpha(80)
        painter.setBrush(QBrush(outer))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(cx - r - 3), int(cy - r - 3),
                            int((r + 3) * 2), int((r + 3) * 2))

    # Filled circle with radial gradient
    grad = QRadialGradient(cx, cy - r * 0.3, r)
    light = QColor(accent)
    light.setAlpha(255)
    dark  = QColor(accent)
    dark.setAlphaF(0.65)
    grad.setColorAt(0, light)
    grad.setColorAt(1, dark)

    painter.setBrush(QBrush(grad))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))
    painter.end()

    return QIcon(px)


# ---------------------------------------------------------------------------
# Background pipeline thread
# ---------------------------------------------------------------------------
class PipelineThread(QThread):
    """
    Runs EyeTracker + KeystrokeMonitor + SlidingWindow in a QThread.
    Emits `result_ready` signal for every completed window.
    """
    result_ready = pyqtSignal(object)  # CognitiveLoadResult

    def __init__(
        self,
        camera_index: int   = 0,
        window_size:  float = 7.0,
        step_size:    float = 2.0,
        no_camera:    bool  = False,
        show_preview: bool  = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.camera_index = camera_index
        self.window_size  = window_size
        self.step_size    = step_size
        self.no_camera    = no_camera
        self.show_preview = show_preview
        self._extractor   = FeatureExtractor()
        self._last_wf     = None   # most recent WindowedFeatures (for labeller)

        self._eye_tracker       = None
        self._keystroke_monitor = None
        self._sliding_window    = None

    @property
    def extractor(self) -> FeatureExtractor:
        return self._extractor

    @property
    def last_wf(self):
        return self._last_wf

    def run(self) -> None:
        import queue as _q
        from cognitive_monitor.acquisition import (
            EyeTracker, KeystrokeMonitor, SlidingWindow
        )

        keystroke_monitor = KeystrokeMonitor()
        keystroke_monitor.start()

        eye_tracker = None
        if not self.no_camera:
            eye_tracker = EyeTracker(camera_index=self.camera_index, target_fps=30, show_preview=self.show_preview)
            eye_tracker.start()

        class _EmptyQueue:
            def get_nowait(self):
                raise _q.Empty

        eye_q = eye_tracker.frame_queue if eye_tracker else _EmptyQueue()

        def _on_window(wf) -> None:
            self._last_wf = wf
            result = self._extractor.predict(wf)
            self.result_ready.emit(result)

        sliding_window = SlidingWindow(
            eye_queue        = eye_q,
            keystroke_queue  = keystroke_monitor.event_queue,
            window_size      = self.window_size,
            step_size        = self.step_size,
            on_window        = _on_window,
        )
        sliding_window.start()

        # Keep thread alive until interrupted
        self.exec()  # Qt event loop for the thread

        # Cleanup
        sliding_window.stop()
        keystroke_monitor.stop()
        if eye_tracker:
            eye_tracker.stop()

    def retrain_model(self, X, y) -> None:
        """Called from the main thread; safe because _extractor is thread-safe."""
        self._extractor.fit(X, y)


# ---------------------------------------------------------------------------
# CognitiveMonitorApp
# ---------------------------------------------------------------------------
class CognitiveMonitorApp(QObject):          # FIX 1: inherit QObject
    """
    Top-level application controller.

    Usage
    -----
    >>> import sys
    >>> from PyQt6.QtWidgets import QApplication
    >>> qapp = QApplication(sys.argv)
    >>> app  = CognitiveMonitorApp(qapp)
    >>> app.start()
    >>> sys.exit(qapp.exec())
    """

    def __init__(
        self,
        qapp:         QApplication,
        camera_index: int   = 0,
        window_size:  float = 7.0,
        step_size:    float = 2.0,
        no_camera:    bool  = False,
        show_preview: bool  = False,
        label_interval_min: float = 10.0,
    ) -> None:
        super().__init__()                   # FIX 2: call QObject.__init__
        self._qapp           = qapp
        self._last_alert_t   = 0.0
        self._current_result: Optional[CognitiveLoadResult] = None
        self._pulse_state    = False

        qapp.setApplicationName("CognitiveMonitor")
        qapp.setQuitOnLastWindowClosed(False)
        qapp.setStyleSheet(theme.APP_STYLESHEET)

        # ---- Tray icon ----
        self._tray = QSystemTrayIcon(
            _make_tray_icon(CognitiveLoadClass.LOW), qapp
        )
        self._tray.setToolTip("CognitiveMonitor — Low Load")
        self._build_tray_menu()
        self._tray.show()

        # ---- Dashboard ----
        self._dashboard = DashboardWindow()

        # ---- Training data ----
        self._store     = TrainingDataStore()
        self._labeller  = LabellingScheduler(self._store, label_interval_min)

        # ---- Pulse timer for critical state ----
        self._pulse_timer = QTimer()
        self._pulse_timer.setInterval(600)
        self._pulse_timer.timeout.connect(self._pulse_tick)

        # ---- Pipeline thread ----
        self._pipeline = PipelineThread(
            camera_index=camera_index,
            window_size=window_size,
            step_size=step_size,
            no_camera=no_camera,
            show_preview=show_preview,
        )
        self._pipeline.result_ready.connect(self._on_result)  # FIX: now works

        # ---- Retrain timer (check every 60 s if new data available) ----
        self._retrain_timer = QTimer()
        self._retrain_timer.setInterval(60_000)
        self._retrain_timer.timeout.connect(self._maybe_retrain)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._pipeline.start()
        self._labeller.start()
        self._retrain_timer.start()
        logger.info("CognitiveMonitorApp started.")

    def stop(self) -> None:
        self._pipeline.quit()
        self._pipeline.wait(3000)
        self._labeller.stop()
        self._tray.hide()
        logger.info("CognitiveMonitorApp stopped.")

    # ------------------------------------------------------------------
    # Tray menu
    # ------------------------------------------------------------------
    def _build_tray_menu(self) -> None:
        menu = QMenu()

        # Current state header (non-interactive)
        self._state_action = QAction("● Low Load — 0", menu)
        self._state_action.setEnabled(False)
        self._state_action.setFont(
            self._state_action.font()   # modified below via stylesheet
        )
        menu.addAction(self._state_action)
        menu.addSeparator()

        open_dash = QAction("📊  Open Dashboard", menu)
        open_dash.triggered.connect(self._open_dashboard)
        menu.addAction(open_dash)

        label_now = QAction("🏷  Label My State Now", menu)
        label_now.triggered.connect(self._prompt_label)
        menu.addAction(label_now)

        menu.addSeparator()

        quit_action = QAction("✕  Quit", menu)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._tray_activated)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    @pyqtSlot(object)
    def _on_result(self, result: CognitiveLoadResult) -> None:
        self._current_result = result
        cls   = result.load_class
        score = result.score

        # Update tray icon
        is_critical = cls == CognitiveLoadClass.CRITICAL
        if is_critical and not self._pulse_timer.isActive():
            self._pulse_timer.start()
        elif not is_critical and self._pulse_timer.isActive():
            self._pulse_timer.stop()
            self._tray.setIcon(_make_tray_icon(cls))

        if not is_critical:
            self._tray.setIcon(_make_tray_icon(cls))

        self._tray.setToolTip(
            f"CognitiveMonitor — {theme.label_for(cls)}  ({score:.0f}/100)"
        )
        self._state_action.setText(
            f"● {theme.label_for(cls)}  —  {score:.0f}"
        )

        # Push to dashboard (thread-safe)
        self._dashboard.update_result(result)

        # Update labeller's feature snapshot
        if self._pipeline.last_wf is not None:
            self._labeller.update_features(self._pipeline.last_wf)

        # Intervention check
        if result.needs_intervention:
            self._maybe_show_intervention(result)

    def _maybe_show_intervention(self, result: CognitiveLoadResult) -> None:
        now = time.time()
        if now - self._last_alert_t < INTERVENTION_COOLDOWN_S:
            return
        self._last_alert_t = now
        popup = InterventionPopup(
            load_class    = result.load_class,
            score         = result.score,
            auto_close_ms = 30_000,
        )
        popup.show_at_bottom_right()
        logger.info("Intervention popup shown (%s, score=%.1f)",
                    result.load_class.value, result.score)

    def _pulse_tick(self) -> None:
        self._pulse_state = not self._pulse_state
        self._tray.setIcon(
            _make_tray_icon(CognitiveLoadClass.CRITICAL, pulse=self._pulse_state)
        )

    def _open_dashboard(self) -> None:
        self._dashboard.show()
        self._dashboard.raise_()
        self._dashboard.activateWindow()

    def _prompt_label(self) -> None:
        from .labeller import LabellerPopup
        fv    = (self._pipeline.last_wf.feature_vector.copy()
                 if self._pipeline.last_wf else None)
        popup = LabellerPopup(feature_vector=fv)
        popup.labelled.connect(
            lambda fv, li: (
                self._store.append(fv, li),
                self._maybe_retrain()
            )
        )
        popup.show()

    def _maybe_retrain(self) -> None:
        data = self._store.load()
        if data is None:
            return
        X, y = data
        n_classes = len(set(y.tolist()))
        if len(X) >= 50 and n_classes >= 2:
            logger.info("Retraining ML model on %d samples…", len(X))
            self._pipeline.retrain_model(X, y)

    def _tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._open_dashboard()

    def _quit(self) -> None:
        self.stop()
        self._qapp.quit()
