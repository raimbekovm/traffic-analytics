# Traffic Analytics Pipeline

Real-time vehicle detection, tracking, and counting system for traffic camera feeds.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This project demonstrates a complete computer vision pipeline for traffic analytics:

- **Vehicle Detection** using YOLOv8/v11 (ultralytics)
- **Multi-Object Tracking** with ByteTrack/BoT-SORT
- **Line Crossing Counter** with IN/OUT direction detection
- **Real-time Visualization** of tracks, trajectories, and statistics
- **Multiple Video Sources** support (files, RTSP, HTTP streams, YouTube)

Built as a demonstration for traffic monitoring systems similar to "Safe City" projects.

## Features

| Feature | Description |
|---------|-------------|
| Vehicle Detection | Detects cars, buses, trucks, motorcycles |
| Object Tracking | Persistent ID assignment across frames |
| Direction Counting | Counts vehicles crossing configurable lines |
| Trajectory Visualization | Shows movement paths of tracked vehicles |
| Multiple Sources | Local files, RTSP cameras, HTTP streams |
| Interactive Setup | Click-to-define counting lines |
| Event Logging | JSON export of all crossing events |
| Configurable | YAML-based configuration |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Video Source  │────▶│    YOLO + Track │────▶│  Line Counter   │
│  (RTSP/File/...)│     │   (ByteTrack)   │     │   (IN/OUT)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │                        │
                                ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   Visualizer    │     │  Event Logger   │
                        │ (boxes, trails) │     │    (JSON)       │
                        └─────────────────┘     └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA (optional, for GPU acceleration)
- MPS (Apple Silicon, automatic)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/traffic-analytics.git
cd traffic-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (automatic on first run)
# Models: yolov8n (fast), yolov8s (balanced), yolov8m/l/x (accurate)
```

## Quick Start

### Basic Usage

```bash
# Run with video file
python main.py --source path/to/video.mp4 --show

# Run with RTSP stream
python main.py --source "rtsp://camera_ip:554/stream" --source-type rtsp --show

# Run with YouTube video
python main.py --source "https://www.youtube.com/watch?v=VIDEO_ID" --source-type youtube --show
```

### Configuration

Edit `config.yaml` to customize:

```yaml
# Video source
video:
  source: "demo/traffic.mp4"
  source_type: "file"
  output_path: "output/result.mp4"
  show_display: true

# Detection
detection:
  model: "yolov8s"
  confidence: 0.5
  device: "mps"  # cpu, cuda, or mps

# Tracking
tracking:
  tracker: "bytetrack"  # or botsort

# Counting lines
counting:
  enabled: true
  lines:
    - name: "Main Road"
      start: [100, 400]
      end: [1180, 400]
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--source`, `-s` | Video source path or URL |
| `--source-type`, `-t` | Source type: file, rtsp, http, youtube |
| `--output`, `-o` | Output video path |
| `--model` | YOLO model: yolov8n/s/m/l/x |
| `--device` | Inference device: cpu, cuda, mps |
| `--confidence` | Detection confidence threshold |
| `--show` | Enable display window |
| `--no-show` | Disable display window |
| `--config`, `-c` | Config file path |

## Project Structure

```
traffic-analytics/
├── main.py                 # Main pipeline entry point
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── video_source.py     # Video input handling
│   ├── detector.py         # YOLO detection wrapper
│   ├── tracker.py          # Multi-object tracking
│   ├── counter.py          # Line crossing counter
│   └── visualizer.py       # Visualization utilities
├── output/                 # Output videos and logs
└── demo/                   # Sample videos
```

## Module Details

### Video Source (`src/video_source.py`)

Handles multiple video input types with automatic reconnection for streams:

```python
from src.video_source import VideoSource

source = VideoSource(
    source="rtsp://camera:554/stream",
    source_type="rtsp",
    frame_skip=2  # Process every 2nd frame
)

with source:
    for frame_info in source.frames():
        process(frame_info.frame)
```

### Tracker (`src/tracker.py`)

Combines YOLO detection with ByteTrack/BoT-SORT tracking:

```python
from src.tracker import VehicleTracker

tracker = VehicleTracker(
    model_name="yolov8s",
    tracker="bytetrack",
    device="cuda"
)

result = tracker.track(frame)
for track in result.tracks:
    print(f"ID: {track.track_id}, Class: {track.class_name}")
```

### Counter (`src/counter.py`)

Counts vehicles crossing defined lines with direction detection:

```python
from src.counter import LineCrossingCounter

counter = LineCrossingCounter(
    lines=[{
        "name": "Main Line",
        "start": [100, 400],
        "end": [1100, 400]
    }],
    in_direction=(0, 1)  # Down = IN
)

events = counter.update(tracks)
stats = counter.get_stats()
# {'Line 1': {'count_in': 15, 'count_out': 12, ...}}
```

## Output Format

### Events JSON

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "total_frames": 5000,
  "total_counts": {"in": 45, "out": 38, "total": 83},
  "events": [
    {
      "track_id": 5,
      "line_name": "Main Road",
      "direction": "in",
      "class_name": "car",
      "timestamp": "2024-01-15T10:25:30",
      "frame_id": 1250
    }
  ]
}
```

## Performance

| Model | Device | Resolution | FPS |
|-------|--------|------------|-----|
| YOLOv8n | CPU | 640x480 | ~15 |
| YOLOv8n | MPS (M1) | 640x480 | ~45 |
| YOLOv8s | CUDA | 1280x720 | ~60 |
| YOLOv8m | CUDA | 1920x1080 | ~35 |

## Roadmap

- [x] Vehicle detection (YOLOv8)
- [x] Multi-object tracking (ByteTrack)
- [x] Line crossing counter
- [x] Trajectory visualization
- [x] Multiple video sources
- [ ] License plate detection + OCR
- [ ] Speed estimation via homography
- [ ] Traffic violation detection
- [ ] Multi-camera support
- [ ] REST API for integration

## Technical Skills Demonstrated

- **Computer Vision**: YOLO, OpenCV, object detection/tracking
- **Deep Learning**: Model inference, optimization (CUDA/MPS)
- **Software Engineering**: Modular architecture, clean code
- **Video Processing**: Streams, codecs, real-time processing
- **Python**: Type hints, dataclasses, generators, context managers

## References

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Supervision](https://github.com/roboflow/supervision)

## License

MIT License - feel free to use for your projects.

## Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

*Built as a portfolio demonstration for CV Engineer positions in traffic analytics and Smart City projects.*
