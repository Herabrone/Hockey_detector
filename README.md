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

You can override the input source at launch time without editing config:
```powershell
python -m app.main --config config/config.yaml --video-source 0
python -m app.main --config config/config.yaml --video-source assets/Game1.mp4
```

If you only want terminal output and no preview window:
```powershell
python -m app.main --config config/config.yaml --video-source assets/Game1.mp4 --no-window
```

## Test with Game1.mp4
```powershell
python -m app.main --config config/test_game.yaml
```

You can also run the same file test without the test config:
```powershell
python -m app.main --config config/config.yaml --video-source assets/Game1.mp4
```

If `assets/goal_template.png` exists, the app will print lines like:
```text
[GOAL] team=unknown score=0.842 | 14:32:07
```

If you add optional team templates in `config/test_game.yaml`, the `team=` value will use the best matching team tag.

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
- `--video-source` overrides the config file at runtime.
- Use `--no-window` when you want console-only testing.
- Keep `relay_mode: print` until relay hardware is connected.
- `config/test_game.yaml` is preconfigured to use `assets/Game1.mp4`.
- Team detection is optional and uses the `team_templates` mapping in config.
