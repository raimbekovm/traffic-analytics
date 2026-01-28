"""
Traffic Analytics Pipeline

A demonstration project for vehicle detection, tracking, and counting
on traffic camera feeds.

Usage:
    python main.py                    # Use default config
    python main.py --config custom.yaml
    python main.py --source video.mp4 # Override source
    python main.py --show             # Enable display window
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from datetime import datetime

import cv2
import yaml

from src.video_source import VideoSource
from src.tracker import VehicleTracker
from src.counter import LineCrossingCounter
from src.visualizer import Visualizer, VideoWriter
from src.calibration import CameraCalibrator, SpeedEstimator
from src.violations import ViolationDetector, draw_solid_lines

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Traffic Analytics Pipeline - Vehicle Detection, Tracking & Counting'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--source', '-s',
        type=str,
        help='Override video source from config'
    )
    parser.add_argument(
        '--source-type', '-t',
        type=str,
        choices=['file', 'rtsp', 'http', 'hls', 'youtube'],
        help='Override source type from config'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Override output video path'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show display window'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Disable display window'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Override YOLO model (e.g., yolov8n, yolov8s, yolov8m)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        help='Override inference device'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        help='Override confidence threshold'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        help='Maximum number of frames to process (for recording clips)'
    )
    return parser.parse_args()


def setup_counting_line_interactive(frame: "np.ndarray") -> list:
    """
    Interactive setup for counting line using mouse clicks.

    Click two points to define the line, press 'q' to finish.
    """
    points = []
    temp_frame = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, temp_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(temp_frame, (x, y), 5, (0, 255, 0), -1)
            if len(points) == 2:
                cv2.line(temp_frame, points[0], points[1], (0, 255, 255), 2)
            cv2.imshow('Setup Counting Line', temp_frame)

    cv2.namedWindow('Setup Counting Line')
    cv2.setMouseCallback('Setup Counting Line', mouse_callback)

    print("\n=== Counting Line Setup ===")
    print("Click two points to define the counting line.")
    print("Press 'r' to reset, 'q' to confirm and continue.\n")

    while True:
        cv2.imshow('Setup Counting Line', temp_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            points = []
            temp_frame = frame.copy()
        elif key == ord('q') and len(points) == 2:
            break

    cv2.destroyWindow('Setup Counting Line')

    return [{
        "name": "Line 1",
        "start": list(points[0]),
        "end": list(points[1]),
        "color": [0, 255, 255]
    }]


def run_pipeline(config: dict, args: argparse.Namespace):
    """Run the main traffic analytics pipeline."""

    # Override config with command line arguments
    video_config = config.get('video', {})
    detection_config = config.get('detection', {})
    tracking_config = config.get('tracking', {})
    counting_config = config.get('counting', {})
    viz_config = config.get('visualization', {})
    output_config = config.get('output', {})
    speed_config = config.get('speed', {})

    if args.source:
        video_config['source'] = args.source
    if args.source_type:
        video_config['source_type'] = args.source_type
    if args.output:
        video_config['output_path'] = args.output
    if args.show:
        video_config['show_display'] = True
    if args.no_show:
        video_config['show_display'] = False
    if args.model:
        detection_config['model'] = args.model
    if args.device:
        detection_config['device'] = args.device
    if args.confidence:
        detection_config['confidence'] = args.confidence

    # Initialize video source
    logger.info(f"Initializing video source: {video_config['source']}")
    video_source = VideoSource(
        source=video_config['source'],
        source_type=video_config.get('source_type', 'file'),
        frame_skip=video_config.get('frame_skip', 1)
    )

    if not video_source.open():
        logger.error("Failed to open video source")
        sys.exit(1)

    logger.info(
        f"Video: {video_source.width}x{video_source.height} @ {video_source.fps:.1f} FPS"
    )

    # Initialize tracker
    logger.info(f"Initializing tracker: {tracking_config.get('tracker', 'bytetrack')}")
    tracker = VehicleTracker(
        model_name=detection_config.get('model', 'yolov8s'),
        tracker=tracking_config.get('tracker', 'bytetrack'),
        confidence=detection_config.get('confidence', 0.5),
        classes=detection_config.get('classes', [2, 3, 5, 7]),
        device=detection_config.get('device', 'cpu'),
        imgsz=detection_config.get('imgsz', 640),
        track_buffer=tracking_config.get('track_buffer', 30),
        trajectory_length=viz_config.get('trajectory_length', 30),
        crop=detection_config.get('crop')
    )

    # Initialize counter
    counting_lines = counting_config.get('lines', [])

    # Interactive line setup if no lines configured and display is enabled
    if not counting_lines and video_config.get('show_display', True):
        success, first_frame = video_source.read()
        if success:
            logger.info("No counting lines configured. Starting interactive setup...")
            counting_lines = setup_counting_line_interactive(first_frame.frame)
            # Reset video to beginning
            video_source.release()
            video_source.open()

    counter = None
    if counting_config.get('enabled', True) and counting_lines:
        logger.info(f"Initializing counter with {len(counting_lines)} lines")
        counter = LineCrossingCounter(
            lines=counting_lines,
            in_direction=tuple(counting_config.get('in_direction', [0, 1]))
        )

    # Initialize speed estimator
    speed_estimator = None
    speed_limit = speed_config.get('speed_limit', 60)
    calibration_file = speed_config.get('calibration_file', 'calibration.json')

    if speed_config.get('enabled', False) and Path(calibration_file).exists():
        try:
            calibrator = CameraCalibrator()
            calibration = calibrator.load_calibration(calibration_file)
            speed_estimator = SpeedEstimator(
                calibration=calibration,
                fps=video_source.fps
            )
            logger.info(f"Speed estimation enabled (limit: {speed_limit} km/h)")
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}")

    # Initialize violation detector
    violation_detector = None
    violations_config = config.get('violations', {})
    solid_lines_config = violations_config.get('solid_lines', [])

    if violations_config.get('enabled', True) and solid_lines_config:
        violation_detector = ViolationDetector(solid_lines_config)
        logger.info(f"Violation detection enabled ({len(solid_lines_config)} solid lines)")

    # Initialize visualizer
    visualizer = Visualizer(
        show_boxes=viz_config.get('show_boxes', True),
        show_trajectories=viz_config.get('show_trajectories', True),
        trajectory_length=viz_config.get('trajectory_length', 30),
        show_ids=viz_config.get('show_ids', True),
        show_labels=viz_config.get('show_labels', True),
        show_confidence=viz_config.get('show_confidence', False),
        show_fps=viz_config.get('show_fps', True),
        show_stats=viz_config.get('show_stats', True),
        box_thickness=viz_config.get('box_thickness', 2),
        font_scale=viz_config.get('font_scale', 0.6)
    )

    # Initialize video writer
    video_writer = None
    output_path = video_config.get('output_path')
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        video_writer = VideoWriter(
            output_path=output_path,
            fps=video_source.fps,
            width=video_source.width,
            height=video_source.height
        )
        logger.info(f"Output video: {output_path}")

    # Events storage
    all_events = []

    # Display window
    window_name = "Traffic Analytics"
    show_display = video_config.get('show_display', True)

    if show_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    logger.info("Starting pipeline...")
    print("\n" + "="*50)
    print("Traffic Analytics Pipeline Running")
    print("="*50)
    print("Press 'q' to quit, 's' to save snapshot")
    print("="*50 + "\n")

    frame_count = 0
    try:
        while True:
            # Read frame
            success, frame_info = video_source.read()
            if not success:
                logger.info("End of video stream")
                break

            frame = frame_info.frame
            frame_id = frame_info.frame_id
            frame_count += 1

            # Check max frames limit
            if args.max_frames and frame_count >= args.max_frames:
                logger.info(f"Reached max frames limit: {args.max_frames}")
                break

            # Run tracking
            tracking_result = tracker.track(frame, frame_id)

            # Estimate speeds
            speeds = {}
            if speed_estimator:
                for track in tracking_result.tracks:
                    speed = speed_estimator.estimate_speed(
                        track_id=track.track_id,
                        trajectory=track.trajectory,
                        frame_skip=video_config.get('frame_skip', 1)
                    )
                    if speed is not None:
                        speeds[track.track_id] = speed

            # Count by class (vehicles currently on screen)
            class_counts = {}
            for track in tracking_result.tracks:
                class_name = track.class_name
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Update counter
            counting_stats = None
            total_counts = {"in": 0, "out": 0, "total": 0}

            if counter:
                events = counter.update(tracking_result.tracks, frame_id)
                counting_stats = counter.get_stats()
                total_counts = counter.get_total_counts()

                # Log events
                for event in events:
                    all_events.append(event.to_dict())
                    logger.info(
                        f"[{event.direction.value.upper()}] "
                        f"Vehicle {event.track_id} ({event.class_name}) "
                        f"crossed {event.line_name}"
                    )

            # Check for violations (crossing solid lines)
            if violation_detector:
                violations = violation_detector.check_violations(tracking_result.tracks, frame_id)

            # Render visualization
            output_frame = visualizer.render(
                frame=frame,
                tracks=tracking_result.tracks,
                counting_lines=counter.counting_lines if counter else None,
                counting_stats=counting_stats,
                total_counts=total_counts,
                speeds=speeds if speed_estimator else None,
                speed_limit=speed_limit if speed_estimator else None,
                class_counts=class_counts
            )

            # Draw solid lines and violation counter
            if violation_detector:
                output_frame = draw_solid_lines(
                    output_frame,
                    violation_detector.solid_lines,
                    violation_detector.get_violation_count()
                )

            # Write output video
            if video_writer:
                video_writer.write(output_frame)

            # Display
            if show_display:
                cv2.imshow(window_name, output_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    logger.info("User requested quit")
                    break
                elif key == ord('s'):
                    # Save snapshot
                    snapshot_path = f"output/snapshot_{frame_id}.jpg"
                    Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(snapshot_path, output_frame)
                    logger.info(f"Saved snapshot: {snapshot_path}")

            # Progress logging
            if frame_count % 100 == 0:
                logger.info(
                    f"Processed {frame_count} frames | "
                    f"Tracks: {len(tracking_result)} | "
                    f"Total count: {total_counts['total']}"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        # Cleanup
        video_source.release()
        if video_writer:
            video_writer.release()
        if show_display:
            cv2.destroyAllWindows()

        # Save events
        if output_config.get('save_events', True) and all_events:
            events_path = output_config.get('events_path', 'output/events.json')
            Path(events_path).parent.mkdir(parents=True, exist_ok=True)
            with open(events_path, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "total_frames": frame_count,
                    "total_counts": total_counts,
                    "events": all_events
                }, f, indent=2)
            logger.info(f"Saved events: {events_path}")

        # Print summary
        print("\n" + "="*50)
        print("Pipeline Summary")
        print("="*50)
        print(f"Frames processed: {frame_count}")
        print(f"Total vehicles counted: {total_counts['total']}")
        print(f"  - IN:  {total_counts['in']}")
        print(f"  - OUT: {total_counts['out']}")
        if output_path:
            print(f"Output video: {output_path}")
        print("="*50 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    logger.info(f"Loaded config: {config_path}")

    # Run pipeline
    run_pipeline(config, args)


if __name__ == '__main__':
    main()
