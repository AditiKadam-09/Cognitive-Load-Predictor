"""
ui/theme.py
===========
Design system for CognitiveMonitor.

Aesthetic direction: "biometric terminal" — the visual language of a
medical-grade vital-signs monitor. Deep charcoal substrates, razor-thin
amber rule-lines, state-reactive accent colours (green → amber → coral →
crimson), and a monospace readout font for numeric data.

All colours defined here; every widget imports from this module so a
global theme change needs touching exactly one file.
"""

from __future__ import annotations
from cognitive_monitor.analysis.feature_extractor import CognitiveLoadClass

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG_DEEP      = "#0B0E14"   # application background
BG_SURFACE   = "#111620"   # card / panel surface
BG_RAISED    = "#1A2030"   # elevated element (button, badge)
BG_OVERLAY   = "#0D1018E0" # translucent overlay (230/255 alpha)

BORDER       = "#1E2840"   # default border
BORDER_GLOW  = "#2A3A60"   # hovered / active border

TEXT_PRIMARY  = "#E8EDF5"
TEXT_SECONDARY = "#6B7A99"
TEXT_DIM      = "#3A4560"

# Load-state accent colours
ACCENT_LOW      = "#2DD4BF"   # teal  — calm
ACCENT_MEDIUM   = "#F5A623"   # amber — alert
ACCENT_HIGH     = "#F97316"   # orange — warning
ACCENT_CRITICAL = "#EF4444"   # red   — danger
ACCENT_DEFAULT  = "#3B82F6"   # blue  — neutral / idle

# Glow shadows (for QGraphicsDropShadow)
GLOW_LOW      = "#2DD4BF"
GLOW_MEDIUM   = "#F5A623"
GLOW_HIGH     = "#F97316"
GLOW_CRITICAL = "#EF4444"

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_MONO   = "JetBrains Mono, Cascadia Code, Consolas, monospace"
FONT_DISPLAY= "Syne, Outfit, Segoe UI, sans-serif"
FONT_BODY   = "DM Sans, Nunito, Segoe UI, sans-serif"

# ---------------------------------------------------------------------------
# State → accent mapping helpers
# ---------------------------------------------------------------------------
_STATE_ACCENTS = {
    CognitiveLoadClass.LOW:      ACCENT_LOW,
    CognitiveLoadClass.MEDIUM:   ACCENT_MEDIUM,
    CognitiveLoadClass.HIGH:     ACCENT_HIGH,
    CognitiveLoadClass.CRITICAL: ACCENT_CRITICAL,
}

_STATE_GLOWS = {
    CognitiveLoadClass.LOW:      GLOW_LOW,
    CognitiveLoadClass.MEDIUM:   GLOW_MEDIUM,
    CognitiveLoadClass.HIGH:     GLOW_HIGH,
    CognitiveLoadClass.CRITICAL: GLOW_CRITICAL,
}

_STATE_LABELS = {
    CognitiveLoadClass.LOW:      "LOW",
    CognitiveLoadClass.MEDIUM:   "MEDIUM",
    CognitiveLoadClass.HIGH:     "HIGH LOAD",
    CognitiveLoadClass.CRITICAL: "CRITICAL",
}


def accent_for(cls: CognitiveLoadClass) -> str:
    return _STATE_ACCENTS.get(cls, ACCENT_DEFAULT)

def glow_for(cls: CognitiveLoadClass) -> str:
    return _STATE_GLOWS.get(cls, ACCENT_DEFAULT)

def label_for(cls: CognitiveLoadClass) -> str:
    return _STATE_LABELS.get(cls, "UNKNOWN")


# ---------------------------------------------------------------------------
# Global QSS stylesheet (applied to QApplication)
# ---------------------------------------------------------------------------
APP_STYLESHEET = f"""
QWidget {{
    background-color: {BG_DEEP};
    color: {TEXT_PRIMARY};
    font-family: {FONT_BODY};
    font-size: 13px;
    border: none;
    outline: none;
}}

QToolTip {{
    background: {BG_RAISED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_GLOW};
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
}}

QMenu {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER_GLOW};
    border-radius: 8px;
    padding: 4px 0px;
}}
QMenu::item {{
    padding: 6px 20px 6px 16px;
    color: {TEXT_PRIMARY};
    font-size: 13px;
}}
QMenu::item:selected {{
    background-color: {BG_RAISED};
    color: {ACCENT_DEFAULT};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 12px;
}}

QScrollBar:vertical {{
    background: {BG_SURFACE};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_GLOW};
    border-radius: 3px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""
