"""
Multi-object tracker using ByteTrack/BoT-SORT.
Integrates with ultralytics built-in tracking.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Single tracked object."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Get bottom center point (for ground plane tracking)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    def update_trajectory(self, max_length: int = 30):
        """Update trajectory with current position."""
        self.trajectory.append(self.center)
        if len(self.trajectory) > max_length:
            self.trajectory = self.trajectory[-max_length:]


@dataclass
class TrackingResult:
    """Container for all tracks in a frame."""
    tracks: List[Track] = field(default_factory=list)
    frame_id: int = 0

    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self):
        return iter(self.tracks)

    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    @property
    def track_ids(self) -> List[int]:
        """Get all track IDs."""
        return [t.track_id for t in self.tracks]


class VehicleTracker:
    """
    Multi-object tracker for vehicles.

    Uses ultralytics built-in tracking (ByteTrack or BoT-SORT).
    """

    # COCO class names for vehicles
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_name: str = "yolov8s",
        tracker: str = "bytetrack",
        confidence: float = 0.5,
        classes: Optional[List[int]] = None,
        device: str = "cpu",
        imgsz: int = 640,
        track_buffer: int = 30,
        trajectory_length: int = 30
    ):
        """
        Initialize tracker.

        Args:
            model_name: YOLO model name
            tracker: Tracker type ("bytetrack" or "botsort")
            confidence: Confidence threshold
            classes: List of class IDs to track
            device: Inference device
            imgsz: Input image size
            track_buffer: Frames to keep lost tracks
            trajectory_length: Max trajectory points to store
        """
        self.model_name = model_name
        self.tracker_type = tracker
        self.confidence = confidence
        self.classes = classes or list(self.VEHICLE_CLASSES.keys())
        self.device = device
        self.imgsz = imgsz
        self.track_buffer = track_buffer
        self.trajectory_length = trajectory_length

        self.model = None
        self._trajectories: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self._class_map: Dict[int, Tuple[int, str]] = {}  # track_id -> (class_id, class_name)

        self._load_model()

    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO

            model_path = self.model_name
            if not model_path.endswith('.pt'):
                model_path += '.pt'

            logger.info(f"Loading tracking model: {model_path}")
            self.model = YOLO(model_path)
            logger.info(f"Tracker initialized: {self.tracker_type}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def track(self, frame: np.ndarray, frame_id: int = 0) -> TrackingResult:
        """
        Run tracking on a frame.

        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier

        Returns:
            TrackingResult containing all tracks
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Run tracking
        results = self.model.track(
            frame,
            conf=self.confidence,
            classes=self.classes,
            device=self.device,
            imgsz=self.imgsz,
            tracker=f"{self.tracker_type}.yaml",
            persist=True,
            verbose=False
        )[0]

        # Parse results
        tracks = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for bbox, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                class_name = self.VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")

                # Store class info for this track
                self._class_map[track_id] = (cls_id, class_name)

                # Update trajectory
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                self._trajectories[track_id].append(center)

                # Limit trajectory length
                if len(self._trajectories[track_id]) > self.trajectory_length:
                    self._trajectories[track_id] = self._trajectories[track_id][-self.trajectory_length:]

                track = Track(
                    track_id=int(track_id),
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=class_name,
                    trajectory=list(self._trajectories[track_id])
                )
                tracks.append(track)

        return TrackingResult(tracks=tracks, frame_id=frame_id)

    def reset(self):
        """Reset tracker state."""
        self._trajectories.clear()
        self._class_map.clear()

        # Reload model to reset internal tracker state
        self._load_model()

    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """Get trajectory for a specific track."""
        return list(self._trajectories.get(track_id, []))

    def get_all_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """Get all trajectories."""
        return dict(self._trajectories)

    @property
    def active_track_count(self) -> int:
        """Get number of currently tracked objects."""
        return len(self._trajectories)


class TrackingAnalytics:
    """Analytics and statistics for tracking results."""

    def __init__(self):
        self.total_unique_tracks = set()
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.frame_count = 0

    def update(self, result: TrackingResult):
        """Update analytics with new tracking result."""
        self.frame_count += 1

        for track in result.tracks:
            if track.track_id not in self.total_unique_tracks:
                self.total_unique_tracks.add(track.track_id)
                self.class_counts[track.class_name] += 1

    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            "total_unique_vehicles": len(self.total_unique_tracks),
            "by_class": dict(self.class_counts),
            "frames_processed": self.frame_count
        }
