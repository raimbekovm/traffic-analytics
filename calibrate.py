#!/usr/bin/env python3
"""
Interactive camera calibration script.

Usage:
    python calibrate.py                    # Use default config
    python calibrate.py --source video.mp4 # Specific source
    python calibrate.py --output calib.json
"""

import argparse
import yaml
from pathlib import Path

from src.video_source import VideoSource
from src.calibration import CameraCalibrator


def main():
    parser = argparse.ArgumentParser(description="Camera Calibration Tool")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file")
    parser.add_argument("--source", "-s", help="Video source override")
    parser.add_argument("--source-type", "-t", default="hls", help="Source type")
    parser.add_argument("--output", "-o", default="calibration.json", help="Output file")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    video_config = config.get("video", {})
    source = args.source or video_config.get("source")
    source_type = video_config.get("source_type", args.source_type)

    print(f"Opening video source: {source}")

    # Open video
    video = VideoSource(source=source, source_type=source_type)
    if not video.open():
        print("Failed to open video source")
        return 1

    # Get first frame
    success, frame_info = video.read()
    if not success:
        print("Failed to read frame")
        return 1

    video.release()

    # Run calibration
    calibrator = CameraCalibrator()
    calibration = calibrator.calibrate_interactive(frame_info.frame)

    if calibration:
        calibrator.save_calibration(args.output)
        print(f"\nCalibration saved to: {args.output}")
        print("\nTo use in main.py, add to config.yaml:")
        print(f'  calibration_file: "{args.output}"')
        return 0
    else:
        print("Calibration cancelled")
        return 1


if __name__ == "__main__":
    exit(main())
