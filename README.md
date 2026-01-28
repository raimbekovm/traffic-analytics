# Traffic Analytics Pipeline

Real-time vehicle detection, tracking, and traffic violation detection system.

## Features

| Module | Description |
|--------|-------------|
| Detection | YOLO26 with ROI cropping |
| Tracking | ByteTrack with class stabilization |
| Counting | Line crossing with IN/OUT direction |
| Speed | Homography calibration, EMA smoothing |
| Violations | Solid line crossing detection |

## Usage

```bash
pip install -r requirements.txt

python main.py --config config.yaml --show
python main.py --source video.mp4 --output output/result.mp4
python main.py --max-frames 600 --no-show  # Record 30 sec
```

## Configuration

See `config.yaml` for all options:
- Video source (file, RTSP, HLS, HTTP)
- Detection model and confidence
- ROI crop settings
- Counting lines
- Speed calibration
- Violation zones

## Tools

```bash
python calibrate.py         # Camera calibration for speed estimation
python draw_solid_line.py   # Draw violation detection zones
```

## Output

- `output/result.mp4` — Annotated video
- `output/events.json` — Crossing events log

## Tech Stack

Python, OpenCV, YOLO26 (ultralytics), ByteTrack
