"""
main.py  (Phase 2 — full PyQt6 integration)
============================================
Entry point for CognitiveMonitor.

Start modes
-----------
  python main.py                   # full mode (webcam + keyboard + tray UI)
  python main.py --no-camera       # keyboard-only
  python main.py --preview         # show OpenCV debug window
  python main.py --headless        # no Qt UI — prints to stdout (CI / debug)
"""

from __future__ import annotations
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def run_headless(args) -> None:
    import signal, time, queue
    from cognitive_monitor.acquisition import EyeTracker, KeystrokeMonitor, SlidingWindow
    from cognitive_monitor.analysis    import FeatureExtractor
    from cognitive_monitor.utils       import CPUGuard, FrameSkipController, patch_eye_tracker_with_cpu_guard

    keystroke_monitor = KeystrokeMonitor()
    keystroke_monitor.start()
    cpu_ctrl  = FrameSkipController()
    cpu_guard = CPUGuard(cpu_ctrl)
    cpu_guard.start()

    eye_tracker = None
    if not args.no_camera:
        eye_tracker = EyeTracker(camera_index=args.camera, target_fps=30, show_preview=args.preview)
        patch_eye_tracker_with_cpu_guard(eye_tracker, cpu_ctrl)
        eye_tracker.start()

    extractor = FeatureExtractor()

    class _EmptyQueue:
        def get_nowait(self):
            raise queue.Empty

    def _print_result(r) -> None:
        ts = time.strftime("%H:%M:%S", time.localtime(r.timestamp))
        logger.info("[%s] %s  score=%.1f  conf=%.2f  source=%s", ts, r.load_class.value, r.score, r.confidence, r.source)
        if r.needs_intervention:
            logger.warning("INTERVENTION: %s", r.intervention_message)

    sw = SlidingWindow(
        eye_queue       = eye_tracker.frame_queue if eye_tracker else _EmptyQueue(),
        keystroke_queue = keystroke_monitor.event_queue,
        window_size     = args.window,
        step_size       = args.step,
        on_window       = lambda wf: _print_result(extractor.predict(wf)),
    )
    sw.start()

    def _shutdown(sig, frame):
        sw.stop(); keystroke_monitor.stop()
        if eye_tracker: eye_tracker.stop()
        cpu_guard.stop(); sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    logger.info("CognitiveMonitor running (headless). Ctrl+C to exit.")
    while True:
        time.sleep(1)


def run_gui(args) -> None:
    from PyQt6.QtWidgets import QApplication
    from cognitive_monitor.ui import CognitiveMonitorApp
    qapp = QApplication(sys.argv)
    app  = CognitiveMonitorApp(
        qapp=qapp, camera_index=args.camera,
        window_size=args.window, step_size=args.step,
        no_camera=args.no_camera, show_preview=args.preview,
        label_interval_min=args.label_interval,
    )
    app.start()
    sys.exit(qapp.exec())


def main() -> None:
    parser = argparse.ArgumentParser(description="CognitiveMonitor")
    parser.add_argument("--no-camera",      action="store_true")
    parser.add_argument("--preview",        action="store_true")
    parser.add_argument("--headless",       action="store_true")
    parser.add_argument("--camera",         type=int,   default=0)
    parser.add_argument("--window",         type=float, default=7.0)
    parser.add_argument("--step",           type=float, default=2.0)
    parser.add_argument("--label-interval", type=float, default=10.0)
    args = parser.parse_args()
    if args.headless:
        run_headless(args)
    else:
        run_gui(args)

if __name__ == "__main__":
    main()
