# Hockey Detector MVP

Minimal implementation for detecting the broadcast `GOAL` overlay and triggering a light relay.

## What is included now
- Config-driven feed loop
- Frame resize and ROI crop
- Template matching score for GOAL overlay
- Cooldown gate to avoid repeat triggers
- Relay abstraction (`print` mode and `usb_serial` mode)
- Template capture helper script

## Repo layout
- `app/main.py` - main runtime loop
- `app/config.py` - config schema and loader
- `app/video_source.py` - video source wrapper
- `app/detector.py` - template matching detector
- `app/gate.py` - cooldown gate
- `app/relay.py` - relay implementations
- `config/config.yaml` - runtime settings
- `scripts/capture_template.py` - save ROI as template image
- `tests/test_gate.py` - cooldown tests

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```powershell
python -m app.main --config config/config.yaml
```

## Capture template
1. Set `video_source` and `roi` in `config/config.yaml`.
2. Run:
```powershell
python -m scripts.capture_template --config config/config.yaml --output assets/goal_template.png
```
3. Press `s` when GOAL overlay is visible.

## Notes
- For recorded-video tuning, set `video_source` to a file path.
- For live capture card, set `video_source` to camera index string such as `"0"`.
- Keep `relay_mode: print` until relay hardware is connected.
