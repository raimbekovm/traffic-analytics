"""
Camera calibration module for speed estimation.
Uses homography to convert pixel coordinates to real-world meters.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Stores calibration data."""
    image_points: List[Tuple[int, int]]  # 4 points in pixels
    world_points: List[Tuple[float, float]]  # 4 points in meters
    homography_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "image_points": self.image_points,
            "world_points": self.world_points,
            "homography_matrix": self.homography_matrix.tolist() if self.homography_matrix is not None else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationData":
        cal = cls(
            image_points=data["image_points"],
            world_points=data["world_points"]
        )
        if data.get("homography_matrix"):
            cal.homography_matrix = np.array(data["homography_matrix"])
        return cal


class CameraCalibrator:
    """
    Interactive camera calibration for speed estimation.

    User clicks 4 points on the road to define a quadrilateral
    with known real-world dimensions.
    """

    def __init__(self):
        self.points: List[Tuple[int, int]] = []
        self.calibration: Optional[CalibrationData] = None
        self._temp_frame = None
        self._window_name = "Camera Calibration"

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            # Draw point
            cv2.circle(self._temp_frame, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(
                self._temp_frame,
                f"P{len(self.points)}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            # Draw lines between points
            if len(self.points) > 1:
                cv2.line(
                    self._temp_frame,
                    self.points[-2],
                    self.points[-1],
                    (0, 255, 255),
                    2
                )
            if len(self.points) == 4:
                cv2.line(
                    self._temp_frame,
                    self.points[-1],
                    self.points[0],
                    (0, 255, 255),
                    2
                )
            cv2.imshow(self._window_name, self._temp_frame)

    def calibrate_interactive(self, frame: np.ndarray) -> Optional[CalibrationData]:
        """
        Run interactive calibration.

        Args:
            frame: Video frame for calibration

        Returns:
            CalibrationData or None if cancelled
        """
        self.points = []
        self._temp_frame = frame.copy()

        # Instructions
        instructions = [
            "=== CAMERA CALIBRATION ===",
            "Click 4 points forming a rectangle on the road",
            "Points should be: P1(top-left), P2(top-right),",
            "                  P3(bottom-right), P4(bottom-left)",
            "Press 'r' to reset, 'q' to cancel, Enter to confirm"
        ]

        y_offset = 30
        for line in instructions:
            cv2.putText(
                self._temp_frame, line, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            y_offset += 25

        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)
        cv2.imshow(self._window_name, self._temp_frame)

        print("\n" + "="*50)
        print("CAMERA CALIBRATION")
        print("="*50)
        print("Click 4 points on the road forming a rectangle.")
        print("Order: top-left -> top-right -> bottom-right -> bottom-left")
        print("Press 'r' to reset, 'q' to cancel, Enter when done")
        print("="*50 + "\n")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                # Reset
                self.points = []
                self._temp_frame = frame.copy()
                y_offset = 30
                for line in instructions:
                    cv2.putText(
                        self._temp_frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                    y_offset += 25
                cv2.imshow(self._window_name, self._temp_frame)
                print("Reset points")

            elif key == ord('q'):
                cv2.destroyWindow(self._window_name)
                print("Calibration cancelled")
                return None

            elif key == 13 and len(self.points) == 4:  # Enter
                break

        cv2.destroyWindow(self._window_name)

        # Get real-world dimensions from user
        print("\nEnter real-world dimensions:")
        try:
            width = float(input("Width of rectangle (meters, e.g. 3.5 for lane): "))
            height = float(input("Height/Length of rectangle (meters, e.g. 10): "))
        except (ValueError, EOFError):
            print("Invalid input, using defaults: 3.5m x 10m")
            width, height = 3.5, 10.0

        # Define world points (rectangle in meters)
        world_points = [
            (0, 0),           # P1: top-left
            (width, 0),       # P2: top-right
            (width, height),  # P3: bottom-right
            (0, height)       # P4: bottom-left
        ]

        # Calculate homography
        src_pts = np.array(self.points, dtype=np.float32)
        dst_pts = np.array(world_points, dtype=np.float32)

        H, _ = cv2.findHomography(src_pts, dst_pts)

        self.calibration = CalibrationData(
            image_points=self.points,
            world_points=world_points,
            homography_matrix=H
        )

        logger.info(f"Calibration complete: {width}m x {height}m")
        print(f"\nCalibration complete!")
        print(f"Rectangle: {width}m x {height}m")

        return self.calibration

    def save_calibration(self, path: str):
        """Save calibration to JSON file."""
        if self.calibration is None:
            raise ValueError("No calibration data to save")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.calibration.to_dict(), f, indent=2)
        logger.info(f"Saved calibration: {path}")

    def load_calibration(self, path: str) -> CalibrationData:
        """Load calibration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.calibration = CalibrationData.from_dict(data)
        logger.info(f"Loaded calibration: {path}")
        return self.calibration


class SpeedEstimator:
    """
    Estimates vehicle speed using homography transformation.

    Improvements for stable speed readings:
    - EMA (Exponential Moving Average) smoothing
    - Maximum acceleration constraint
    - Minimum trajectory points required
    - Linear regression for distance calculation
    """

    def __init__(
        self,
        calibration: CalibrationData,
        fps: float = 20.0,
        ema_alpha: float = 0.3,
        max_acceleration: float = 15.0,
        min_trajectory_points: int = 5,
        warmup_frames: int = 10
    ):
        """
        Initialize speed estimator.

        Args:
            calibration: Camera calibration data
            fps: Video frames per second
            ema_alpha: EMA smoothing factor (0.1=smooth, 0.5=responsive)
            max_acceleration: Max speed change in km/h per second
            min_trajectory_points: Minimum points before calculating speed
            warmup_frames: Frames before applying smoothing constraints
        """
        if calibration.homography_matrix is None:
            raise ValueError("Calibration has no homography matrix")

        self.H = calibration.homography_matrix
        self.fps = fps
        self.ema_alpha = ema_alpha
        self.ema_alpha_warmup = 0.7  # More responsive during warmup
        self.max_acceleration = max_acceleration
        self.min_trajectory_points = min_trajectory_points
        self.warmup_frames = warmup_frames

        # Track state: {track_id: {"ema_speed": float, "last_raw": float, "frame_count": int}}
        self._track_state: dict = {}

    def pixel_to_world(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates (meters)."""
        pt = np.array([point[0], point[1], 1.0])
        world = self.H @ pt
        world = world / world[2]  # Normalize
        return (float(world[0]), float(world[1]))

    def _calculate_speed_regression(
        self,
        trajectory: List[Tuple[float, float]],
        frame_skip: int
    ) -> Optional[float]:
        """
        Calculate speed using linear regression on world coordinates.
        More stable than point-to-point distance.
        """
        n_points = min(10, len(trajectory))
        recent = trajectory[-n_points:]

        # Convert to world coordinates
        world_points = [self.pixel_to_world(p) for p in recent]

        # Time array (in seconds)
        times = np.array([i * frame_skip / self.fps for i in range(len(world_points))])

        # X and Y coordinates
        xs = np.array([p[0] for p in world_points])
        ys = np.array([p[1] for p in world_points])

        # Linear regression for velocity components
        # vx = dx/dt, vy = dy/dt
        if len(times) < 2:
            return None

        # Fit line: position = velocity * time + offset
        # Using numpy polyfit for slope (velocity)
        try:
            vx = np.polyfit(times, xs, 1)[0]  # slope = velocity
            vy = np.polyfit(times, ys, 1)[0]
        except np.linalg.LinAlgError:
            return None

        # Speed magnitude in m/s -> km/h
        speed_ms = np.sqrt(vx**2 + vy**2)
        speed_kmh = speed_ms * 3.6

        return speed_kmh

    def _is_warmup(self, track_id: int) -> bool:
        """Check if track is still in warmup period."""
        if track_id not in self._track_state:
            return True
        return self._track_state[track_id].get("frame_count", 0) < self.warmup_frames

    def _apply_ema(self, track_id: int, raw_speed: float) -> float:
        """Apply Exponential Moving Average smoothing."""
        if track_id not in self._track_state:
            self._track_state[track_id] = {
                "ema_speed": raw_speed,
                "last_raw": raw_speed,
                "frame_count": 1
            }
            return raw_speed

        state = self._track_state[track_id]
        prev_ema = state["ema_speed"]
        state["frame_count"] = state.get("frame_count", 0) + 1

        # Use higher alpha during warmup for faster convergence
        if self._is_warmup(track_id):
            alpha = self.ema_alpha_warmup
        else:
            alpha = self.ema_alpha

        # EMA formula: new_ema = alpha * raw + (1 - alpha) * prev_ema
        new_ema = alpha * raw_speed + (1 - alpha) * prev_ema

        state["ema_speed"] = new_ema
        state["last_raw"] = raw_speed

        return new_ema

    def _apply_acceleration_limit(self, track_id: int, speed: float, dt: float) -> float:
        """Limit speed change based on physical acceleration constraints."""
        # Skip acceleration limit during warmup
        if self._is_warmup(track_id):
            return speed

        if track_id not in self._track_state:
            return speed

        state = self._track_state[track_id]
        prev_speed = state.get("ema_speed", speed)

        # Max change allowed in this time interval
        max_change = self.max_acceleration * dt

        # Clamp speed change
        speed_change = speed - prev_speed
        if abs(speed_change) > max_change:
            speed = prev_speed + np.sign(speed_change) * max_change

        return speed

    def estimate_speed(
        self,
        track_id: int,
        trajectory: List[Tuple[float, float]],
        frame_skip: int = 1
    ) -> Optional[float]:
        """
        Estimate speed from trajectory with smoothing and constraints.

        Args:
            track_id: Vehicle track ID
            trajectory: List of pixel coordinates
            frame_skip: Frames between trajectory points

        Returns:
            Speed in km/h or None if insufficient data
        """
        # Require minimum trajectory points
        if len(trajectory) < self.min_trajectory_points:
            return None

        # Calculate raw speed using linear regression
        raw_speed = self._calculate_speed_regression(trajectory, frame_skip)

        if raw_speed is None:
            return None

        # Clamp to reasonable range (0-200 km/h)
        raw_speed = max(0, min(200, raw_speed))

        # Time interval for acceleration limit
        dt = frame_skip / self.fps

        # Apply acceleration limit first (before EMA)
        constrained_speed = self._apply_acceleration_limit(track_id, raw_speed, dt)

        # Apply EMA smoothing
        smooth_speed = self._apply_ema(track_id, constrained_speed)

        return round(smooth_speed, 1)

    def get_speed(self, track_id: int) -> Optional[float]:
        """Get last known smoothed speed for a track."""
        if track_id in self._track_state:
            return self._track_state[track_id].get("ema_speed")
        return None

    def clear_track(self, track_id: int):
        """Clear speed history for a track."""
        if track_id in self._track_state:
            del self._track_state[track_id]

    def get_stats(self, track_id: int) -> Optional[dict]:
        """Get speed statistics for debugging."""
        if track_id not in self._track_state:
            return None
        state = self._track_state[track_id]
        return {
            "ema_speed": round(state.get("ema_speed", 0), 1),
            "last_raw": round(state.get("last_raw", 0), 1),
            "frame_count": state.get("frame_count", 0)
        }
