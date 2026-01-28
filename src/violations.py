"""
Traffic violation detection module.
Detects crossing of solid lines (lane violations).
"""

import cv2
import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SolidLine:
    """Definition of a solid line that cannot be crossed."""
    name: str
    points: List[Tuple[int, int]]  # List of points forming the line
    color: Tuple[int, int, int] = (0, 0, 255)  # Red by default
    thickness: int = 3


@dataclass
class Violation:
    """Record of a traffic violation."""
    track_id: int
    line_name: str
    class_name: str
    timestamp: datetime
    frame_id: int
    position: Tuple[float, float]
    violation_type: str = "solid_line_crossing"

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "line_name": self.line_name,
            "class_name": self.class_name,
            "timestamp": self.timestamp.isoformat(),
            "frame_id": self.frame_id,
            "position": [float(self.position[0]), float(self.position[1])],
            "violation_type": self.violation_type
        }


class ViolationDetector:
    """
    Detects traffic violations (crossing solid lines).
    """

    def __init__(self, lines: List[dict]):
        """
        Initialize violation detector.

        Args:
            lines: List of line definitions from config
        """
        self.solid_lines: List[SolidLine] = []
        self._crossed_tracks: Dict[str, set] = {}  # line_name -> set of track_ids
        self.violations: List[Violation] = []

        for line_config in lines:
            line = SolidLine(
                name=line_config['name'],
                points=[tuple(p) for p in line_config['points']],
                color=tuple(line_config.get('color', [0, 0, 255])),
                thickness=line_config.get('thickness', 3)
            )
            self.solid_lines.append(line)
            self._crossed_tracks[line.name] = set()

        logger.info(f"Violation detector initialized with {len(self.solid_lines)} solid lines")

    def _point_to_line_distance(
        self,
        point: Tuple[float, float],
        line_start: Tuple[int, int],
        line_end: Tuple[int, int]
    ) -> float:
        """Calculate distance from point to line segment."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Line vector
        dx = x2 - x1
        dy = y2 - y1

        # If line is a point
        if dx == 0 and dy == 0:
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

        # Parameter t for closest point on line
        t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx*dx + dy*dy)))

        # Closest point on line
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return np.sqrt((x0 - closest_x)**2 + (y0 - closest_y)**2)

    def _crosses_line(
        self,
        trajectory: List[Tuple[float, float]],
        line_points: List[Tuple[int, int]]
    ) -> bool:
        """Check if trajectory crosses the polyline."""
        if len(trajectory) < 2:
            return False

        # Check last movement against each line segment
        p1 = trajectory[-2]
        p2 = trajectory[-1]

        for i in range(len(line_points) - 1):
            l1 = line_points[i]
            l2 = line_points[i + 1]

            if self._segments_intersect(p1, p2, l1, l2):
                return True

        return False

    def _segments_intersect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[int, int],
        p4: Tuple[int, int]
    ) -> bool:
        """Check if two line segments intersect."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def check_violations(
        self,
        tracks: List,
        frame_id: int
    ) -> List[Violation]:
        """
        Check for violations in current frame.

        Args:
            tracks: List of Track objects
            frame_id: Current frame ID

        Returns:
            List of new violations detected
        """
        new_violations = []

        for track in tracks:
            if len(track.trajectory) < 2:
                continue

            for line in self.solid_lines:
                # Skip if already recorded for this track
                if track.track_id in self._crossed_tracks[line.name]:
                    continue

                if self._crosses_line(track.trajectory, line.points):
                    violation = Violation(
                        track_id=track.track_id,
                        line_name=line.name,
                        class_name=track.class_name,
                        timestamp=datetime.now(),
                        frame_id=frame_id,
                        position=track.trajectory[-1]
                    )
                    new_violations.append(violation)
                    self.violations.append(violation)
                    self._crossed_tracks[line.name].add(track.track_id)

                    logger.warning(
                        f"VIOLATION: Vehicle {track.track_id} ({track.class_name}) "
                        f"crossed {line.name}"
                    )

        return new_violations

    def get_violation_count(self) -> int:
        return len(self.violations)


def draw_solid_lines(
    frame: np.ndarray,
    lines: List[SolidLine],
    violation_count: int = 0
) -> np.ndarray:
    """Draw solid lines on frame."""
    output = frame.copy()

    for line in lines:
        pts = np.array(line.points, dtype=np.int32)
        cv2.polylines(output, [pts], isClosed=False, color=line.color, thickness=line.thickness)

        # Draw line name
        if line.points:
            cv2.putText(
                output,
                f"{line.name}",
                (line.points[0][0], line.points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                line.color,
                2
            )

    # Draw violation counter
    if violation_count > 0:
        text = f"VIOLATIONS: {violation_count}"
        cv2.putText(
            output,
            text,
            (output.shape[1] - 250, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return output
