"""
acquisition/keystroke_monitor.py
=================================
Non-intrusive keystroke dynamics monitor built on `pynput`.

Captures:
  • Inter-Key Interval (IKI) — time between consecutive key-down events
  • Hold time (dwell) — duration a key is held (key-down → key-up)
  • Typing speed — characters per second over a rolling window
  • Backspace / delete frequency — error-correction rate
  • Burst coefficient — ratio of very-fast IKIs to mean IKI

Privacy note
------------
Only timing metadata is stored.  Key identities are classified as
"printable", "backspace/delete", "modifier", or "other" — no actual
characters are recorded.

Threading model
---------------
`KeystrokeMonitor` runs pynput listeners on background threads.
Keystroke events are pushed into a `queue.Queue[KeystrokeEvent]` for
the Analysis Engine to consume.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from pynput import keyboard

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations & data classes
# ---------------------------------------------------------------------------
class KeyCategory(Enum):
    PRINTABLE  = auto()   # letters, digits, punctuation
    BACKSPACE  = auto()   # backspace or delete — error-correction keys
    MODIFIER   = auto()   # shift, ctrl, alt, meta
    NAVIGATION = auto()   # arrows, pgup/pgdn, home/end
    OTHER      = auto()   # function keys, enter, escape, etc.


@dataclass
class KeystrokeEvent:
    """
    A single key-press or key-release event.

    Attributes
    ----------
    timestamp   : UNIX time of the event
    event_type  : 'press' or 'release'
    category    : KeyCategory (privacy-preserving classification)
    key_id      : integer hash of the key for matching press↔release pairs
                  (not the character — just used to pair events)
    iki         : Inter-Key Interval from the *previous* press event (seconds).
                  None for the very first press or after a > 5 s pause.
    hold_time   : Key hold duration (press→release).  None until release fires.
    """
    timestamp:  float
    event_type: str           # 'press' | 'release'
    category:   KeyCategory
    key_id:     int           # stable integer ID (hash of pynput key)
    iki:        Optional[float]  = None
    hold_time:  Optional[float]  = None


@dataclass
class KeystrokeWindowStats:
    """Feature vector extracted from a sliding window of keystroke events."""
    window_start:      float
    window_end:        float
    typing_speed_cps:  float   # printable chars per second
    mean_iki:          float   # mean inter-key interval (s)
    std_iki:           float   # IKI standard deviation (s)
    cv_iki:            float   # coefficient of variation = std/mean
    backspace_rate:    float   # backspace events / total key events
    mean_hold:         float   # mean hold time (s)
    burst_coefficient: float   # fraction of IKIs < 0.1 s (fast-burst ratio)
    pause_count:       int     # number of pauses > 1 s within window
    total_keystrokes:  int


# ---------------------------------------------------------------------------
# Categorise pynput key objects (pure function, easy to test)
# ---------------------------------------------------------------------------
_BACKSPACE_KEYS = {
    keyboard.Key.backspace,
    keyboard.Key.delete,
}
_MODIFIER_KEYS = {
    keyboard.Key.shift, keyboard.Key.shift_r,
    keyboard.Key.ctrl,  keyboard.Key.ctrl_r,
    keyboard.Key.alt,   keyboard.Key.alt_r, keyboard.Key.alt_gr,
    keyboard.Key.cmd,   keyboard.Key.cmd_r,
}
_NAVIGATION_KEYS = {
    keyboard.Key.up, keyboard.Key.down,
    keyboard.Key.left, keyboard.Key.right,
    keyboard.Key.home, keyboard.Key.end,
    keyboard.Key.page_up, keyboard.Key.page_down,
}


def categorise_key(key) -> KeyCategory:
    """
    Map a pynput key object to a `KeyCategory`.

    Parameters
    ----------
    key : pynput.keyboard.Key | pynput.keyboard.KeyCode

    Returns
    -------
    KeyCategory
    """
    if key in _BACKSPACE_KEYS:
        return KeyCategory.BACKSPACE
    if key in _MODIFIER_KEYS:
        return KeyCategory.MODIFIER
    if key in _NAVIGATION_KEYS:
        return KeyCategory.NAVIGATION
    # KeyCode objects have a .char attribute for printable characters
    if hasattr(key, "char") and key.char is not None:
        return KeyCategory.PRINTABLE
    return KeyCategory.OTHER


def stable_key_id(key) -> int:
    """
    Return a stable integer ID for a key.
    Uses the vk (virtual key code) when available, otherwise hash().
    """
    if hasattr(key, "vk") and key.vk is not None:
        return int(key.vk)
    return hash(key) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# KeystrokeMonitor
# ---------------------------------------------------------------------------
class KeystrokeMonitor:
    """
    Background keystroke dynamics monitor.

    Usage
    -----
    >>> monitor = KeystrokeMonitor()
    >>> monitor.start()
    >>> evt = monitor.event_queue.get()   # blocks until next event
    >>> monitor.stop()
    """

    # IKI is set to None after a pause longer than this (session boundary)
    IKI_RESET_THRESHOLD_S = 5.0

    def __init__(self, queue_maxsize: int = 512) -> None:
        self.event_queue: queue.Queue[KeystrokeEvent] = queue.Queue(
            maxsize=queue_maxsize
        )
        self._listener: Optional[keyboard.Listener] = None
        self._lock = threading.Lock()

        # State for IKI calculation
        self._last_press_time: Optional[float] = None
        # Map key_id → press timestamp for hold-time calculation
        self._press_times: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the pynput keyboard listener."""
        if self._listener and self._listener.running:
            logger.warning("KeystrokeMonitor already running.")
            return
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()
        logger.info("KeystrokeMonitor started.")

    def stop(self) -> None:
        """Stop the keyboard listener."""
        if self._listener:
            self._listener.stop()
        logger.info("KeystrokeMonitor stopped.")

    # ------------------------------------------------------------------
    # pynput callbacks (called from the listener thread)
    # ------------------------------------------------------------------
    def _on_press(self, key) -> None:
        now    = time.time()
        cat    = categorise_key(key)
        kid    = stable_key_id(key)

        with self._lock:
            # IKI
            iki: Optional[float] = None
            if self._last_press_time is not None:
                gap = now - self._last_press_time
                if gap <= self.IKI_RESET_THRESHOLD_S:
                    iki = gap
            self._last_press_time = now
            self._press_times[kid] = now

        evt = KeystrokeEvent(
            timestamp=now,
            event_type="press",
            category=cat,
            key_id=kid,
            iki=iki,
        )
        self._enqueue(evt)

    def _on_release(self, key) -> None:
        now = time.time()
        cat = categorise_key(key)
        kid = stable_key_id(key)

        with self._lock:
            press_t = self._press_times.pop(kid, None)
            hold    = (now - press_t) if press_t is not None else None

        evt = KeystrokeEvent(
            timestamp=now,
            event_type="release",
            category=cat,
            key_id=kid,
            hold_time=hold,
        )
        self._enqueue(evt)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _enqueue(self, evt: KeystrokeEvent) -> None:
        try:
            self.event_queue.put_nowait(evt)
        except queue.Full:
            # Drop oldest event to make room
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.event_queue.put_nowait(evt)
            except queue.Full:
                pass  # give up for this event
