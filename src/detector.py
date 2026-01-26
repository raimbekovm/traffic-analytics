"""
Vehicle detector using YOLOv8/v11.
Supports detection of: car, bus, truck, motorcycle.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Colors for each class (BGR format)
CLASS_COLORS = {
    "car": (0, 255, 0),       # Green
    "motorcycle": (255, 0, 0), # Blue
    "bus": (0, 165, 255),      # Orange
    "truck": (255, 0, 255),    # Magenta
}


@dataclass
class Detection:
    """Single detection result."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        """Get area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class DetectionResult:
    """Container for all detections in a frame."""
    detections: List[Detection] = field(default_factory=list)
    inference_time: float = 0.0  # milliseconds
    frame_id: int = 0

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self):
        return iter(self.detections)

    def filter_by_class(self, class_names: List[str]) -> 'DetectionResult':
        """Filter detections by class name."""
        filtered = [d for d in self.detections if d.class_name in class_names]
        return DetectionResult(
            detections=filtered,
            inference_time=self.inference_time,
            frame_id=self.frame_id
        )

    def to_numpy(self) -> np.ndarray:
        """Convert detections to numpy array for tracking."""
        if not self.detections:
            return np.empty((0, 6))  # [x1, y1, x2, y2, conf, class_id]

        return np.array([
            [*d.bbox, d.confidence, d.class_id]
            for d in self.detections
        ])


class VehicleDetector:
    """
    YOLO-based vehicle detector.

    Supports YOLOv8 and YOLOv11 models from ultralytics.
    """

    def __init__(
        self,
        model_name: str = "yolov8s",
        confidence: float = 0.5,
        classes: Optional[List[int]] = None,
        device: str = "cpu",
        imgsz: int = 640
    ):
        """
        Initialize detector.

        Args:
            model_name: YOLO model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            confidence: Confidence threshold for detections
            classes: List of class IDs to detect (None = all vehicles)
            device: Inference device ("cpu", "cuda", "mps")
            imgsz: Input image size for inference
        """
        self.model_name = model_name
        self.confidence = confidence
        self.classes = classes or list(VEHICLE_CLASSES.keys())
        self.device = device
        self.imgsz = imgsz
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO

            # Add .pt extension if not present
            model_path = self.model_name
            if not model_path.endswith('.pt'):
                model_path += '.pt'

            logger.info(f"Loading model: {model_path}")
            self.model = YOLO(model_path)

            # Warm up model
            logger.info(f"Warming up model on device: {self.device}")
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(
                dummy,
                device=self.device,
                verbose=False
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> DetectionResult:
        """
        Run detection on a frame.

        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier

        Returns:
            DetectionResult containing all detections
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Run inference
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=self.classes,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False
        )[0]

        # Parse results
        detections = []
        inference_time = results.speed.get('inference', 0)

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for bbox, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name = VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
                detections.append(Detection(
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=class_name
                ))

        return DetectionResult(
            detections=detections,
            inference_time=inference_time,
            frame_id=frame_id
        )

    def detect_batch(
        self,
        frames: List[np.ndarray],
        frame_ids: Optional[List[int]] = None
    ) -> List[DetectionResult]:
        """
        Run detection on multiple frames.

        Args:
            frames: List of input frames
            frame_ids: List of frame identifiers

        Returns:
            List of DetectionResult objects
        """
        if frame_ids is None:
            frame_ids = list(range(len(frames)))

        results = []
        for frame, fid in zip(frames, frame_ids):
            results.append(self.detect(frame, fid))

        return results

    @property
    def class_names(self) -> Dict[int, str]:
        """Get class ID to name mapping."""
        return {cid: VEHICLE_CLASSES.get(cid, f"class_{cid}") for cid in self.classes}

    @staticmethod
    def get_class_color(class_name: str) -> tuple:
        """Get color for a class."""
        return CLASS_COLORS.get(class_name, (255, 255, 255))
