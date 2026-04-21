# CognitiveMonitor

Real-time cognitive load monitoring using **Keystroke Dynamics** +
**Webcam Eye/Face Tracking**. Runs silently in the system tray.
Triggers styled dark-mode pop-ups when load is HIGH or CRITICAL.

---

## Requirements

| Component         | Minimum version |
|-------------------|-----------------|
| Python            | 3.10+           |
| OS                | Windows 10 / macOS 12 / Ubuntu 20.04 |
| Webcam            | Optional (app works without one) |

---

## Installation

### 1 — Clone / unzip

```bash
unzip cognitive_monitor.zip
cd cognitive_monitor
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

On macOS you may also need:
```bash
brew install portaudio   # only if pynput has issues
```

On Linux (headless servers) install the X libraries:
```bash
sudo apt-get install python3-xlib xvfb
```

---

## Running the Application

> **Important:** Run all commands from the **parent folder** that contains
> the `cognitive_monitor/` directory — NOT from inside it.
>
> ```
> Project/                   ← run commands here
> └── cognitive_monitor/
>     ├── main.py
>     └── ...
> ```

### Full GUI mode (system tray + webcam)

```bash
python -m cognitive_monitor.main
```

The app minimises to the **system tray** immediately.
- **Left-click** the tray icon → opens Dashboard
- **Right-click** → context menu (Dashboard / Label / Quit)

### No camera mode (keystroke-only)

```bash
python -m cognitive_monitor.main --no-camera
```

### Headless / debug mode (no Qt, logs to stdout)

```bash
python -m cognitive_monitor.main --headless
python -m cognitive_monitor.main --headless --no-camera
```

### All CLI flags

| Flag                       | Default | Description                              |
|----------------------------|---------|------------------------------------------|
| `--no-camera`              | off     | Disable webcam; keystroke-only mode      |
| `--preview`                | off     | Show OpenCV debug overlay window         |
| `--headless`               | off     | Skip PyQt6 UI; log results to stdout     |
| `--camera INT`             | 0       | Camera device index                      |
| `--window FLOAT`           | 7.0     | Sliding window size in seconds           |
| `--step FLOAT`             | 2.0     | Window step size in seconds              |
| `--label-interval FLOAT`   | 10.0    | Minutes between self-label prompts       |

---

## Running the Test Suite

```bash
# All 24 unit tests (no display needed)
python -m unittest discover -s cognitive_monitor/tests -v

# With pytest (if installed)
python -m pytest cognitive_monitor/tests/ -v
```

---

## Project Structure

```
cognitive_monitor/
├── main.py                         Entry point
├── requirements.txt
├── acquisition/
│   ├── eye_tracker.py              EyeTracker thread (MediaPipe, EAR, PERCLOS)
│   ├── keystroke_monitor.py        KeystrokeMonitor (pynput, IKI, backspace rate)
│   └── sliding_window.py           SlidingWindow aggregator + WindowedFeatures
├── analysis/
│   └── feature_extractor.py        HeuristicScorer + RandomForest fusion model
├── ui/
│   ├── theme.py                    Design tokens (colours, fonts, QSS)
│   ├── tray_app.py                 System tray controller + PipelineThread
│   ├── dashboard.py                Live dashboard (RadialGauge, FeatureBar, Sparkline)
│   ├── intervention_popup.py       Animated dark-mode alert pop-up
│   └── labeller.py                 Self-labelling UI + TrainingDataStore CSV
└── utils/
    └── cpu_monitor.py              CPUGuard + FrameSkipController (≤10% CPU target)
```

---

## Cognitive Load States

| Class    | Score | Tray colour | Behaviour                          |
|----------|-------|-------------|-------------------------------------|
| Low      | 0–29  | Teal        | Silent                             |
| Medium   | 30–59 | Amber       | Silent                             |
| High     | 60–79 | Orange      | Pop-up: 2-min breathing exercise   |
| Critical | 80–100| Red (pulse) | Pop-up: Step away / hydrate        |

Interventions have a **120-second cooldown** between pop-ups.

---

## Machine Learning

The app ships with a **heuristic scorer** active immediately.
After you accumulate ≥ 50 self-labelled windows
(`~/.cognitive_monitor/training_data.csv`), a
**scikit-learn RandomForestClassifier** is trained automatically
and takes over prediction.  The model retrains every 60 seconds
as new labels arrive.

14-feature vector:
`mean_ear · std_ear · blink_rate · gaze_stability · perclos ·
head_roll · face_coverage · typing_speed_cps · mean_iki · std_iki ·
cv_iki · backspace_rate · burst_coefficient · pause_count`

---

## Privacy

- **No characters are ever recorded.** Keys are classified only as
  PRINTABLE / BACKSPACE / MODIFIER / NAVIGATION / OTHER.
- **No video is stored.** Only per-frame EAR and gaze scalars are kept.
- All data stays **local** (`~/.cognitive_monitor/`).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ImportError: pynput` on Linux | `sudo apt install python3-xlib` |
| Camera not found | Try `--camera 1` or `--no-camera` |
| Qt platform error on Linux | `export QT_QPA_PLATFORM=xcb` or install `libxcb-*` |
| High CPU on weak hardware | Increase `--step` to 5.0 |
| No tray icon on GNOME | Install `gnome-shell-extension-appindicator` |
| `ModuleNotFoundError: No module named 'cognitive_monitor'` | Run from the **parent** folder, not inside `cognitive_monitor/` |

---

## Changelog / Bug Fixes (v0.1.1)

| # | File | Fix |
|---|------|-----|
| 1 | `ui/tray_app.py` | `CognitiveMonitorApp` now inherits `QObject` — fixes `TypeError: connect() failed` on startup |
| 2 | `ui/tray_app.py` | `super().__init__()` added to `CognitiveMonitorApp.__init__` |
| 3 | `ui/tray_app.py` | `--preview` flag now correctly propagated through `PipelineThread` → `EyeTracker` in GUI mode |
| 4 | `ui/intervention_popup.py` | `_dismiss()` guarded against double-call (stacking `finished` signal connections) |
| 5 | `ui/labeller.py` | `TrainingDataStore.load()` wrapped in `try/except` — survives corrupt/partial CSV rows |
| 6 | `analysis/feature_extractor.py` | `_ml_predict()` pads probability vector to 4 classes when model trained on fewer; wraps predict in try/except |
| 7 | `requirements.txt` | Added `psutil>=5.9.0` (was missing; required by `CPUGuard`) |
