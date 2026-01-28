"""
Line crossing counter for traffic flow analysis.
Counts vehicles crossing defined lines with direction detection.
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class CrossingDirection(Enum):
    """Direction of line crossing."""
    IN = "in"
    OUT = "out"
    UNKNOWN = "unknown"


@dataclass
class CountingLine:
    """Definition of a counting line."""
    name: str
    start: Tuple[int, int]
    end: Tuple[int, int]
    color: Tuple[int, int, int] = (0, 255, 255)  # Yellow (BGR)
    thickness: int = 2

    @property
    def vector(self) -> np.ndarray:
        """Line direction vector."""
        return np.array([
            self.end[0] - self.start[0],
            self.end[1] - self.start[1]
        ])

    @property
    def normal(self) -> np.ndarray:
        """Normal vector (perpendicular to line)."""
        v = self.vector
        # Rotate 90 degrees: (x, y) -> (-y, x)
        return np.array([-v[1], v[0]])

    @property
    def length(self) -> float:
        """Line length."""
        return np.linalg.norm(self.vector)


@dataclass
class CrossingEvent:
    """Record of a vehicle crossing a line."""
    track_id: int
    line_name: str
    direction: CrossingDirection
    class_name: str
    timestamp: datetime
    frame_id: int
    position: Tuple[float, float]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "track_id": self.track_id,
            "line_name": self.line_name,
            "direction": self.direction.value,
            "class_name": self.class_name,
            "timestamp": self.timestamp.isoformat(),
            "frame_id": self.frame_id,
            "position": [float(self.position[0]), float(self.position[1])]
        }


@dataclass
class CountingStats:
    """Statistics for a counting line."""
    line_name: str
    count_in: int = 0
    count_out: int = 0
    by_class_in: Dict[str, int] = field(default_factory=dict)
    by_class_out: Dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.count_in + self.count_out

    def to_dict(self) -> Dict:
        return {
            "line_name": self.line_name,
            "count_in": self.count_in,
            "count_out": self.count_out,
            "total": self.total,
            "by_class_in": self.by_class_in,
            "by_class_out": self.by_class_out
        }


class LineCrossingCounter:
    """
    Counts vehicles crossing defined lines with direction detection.

    Uses track trajectory to determine crossing direction.
    """

    def __init__(
        self,
        lines: List[Dict],
        in_direction: Tuple[float, float] = (0, 1)
    ):
        """
        Initialize counter.

        Args:
            lines: List of line definitions with keys:
                   - name: Line identifier
                   - start: [x1, y1] start point
                   - end: [x2, y2] end point
                   - color: [B, G, R] color (optional)
            in_direction: Reference direction vector for "IN" classification
                         Default (0, 1) means downward movement is "IN"
        """
        self.lines: List[CountingLine] = []
        self.in_direction = np.array(in_direction)

        for line_def in lines:
            self.lines.append(CountingLine(
                name=line_def["name"],
                start=tuple(line_def["start"]),
                end=tuple(line_def["end"]),
                color=tuple(line_def.get("color", [0, 255, 255]))
            ))

        # Track state
        self._crossed_tracks: Dict[str, Set[int]] = {
            line.name: set() for line in self.lines
        }
        self._stats: Dict[str, CountingStats] = {
            line.name: CountingStats(line_name=line.name) for line in self.lines
        }
        self._events: List[CrossingEvent] = []

    def _point_side_of_line(
        self,
        point: Tuple[float, float],
        line: CountingLine
    ) -> float:
        """
        Determine which side of the line a point is on.

        Returns:
            Positive if on the right side (relative to line direction)
            Negative if on the left side
            Zero if on the line
        """
        # Cross product of line vector and point-to-start vector
        line_vec = line.vector
        point_vec = np.array([
            point[0] - line.start[0],
            point[1] - line.start[1]
        ])
        return np.cross(line_vec, point_vec)

    def _check_line_crossing(
        self,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float],
        line: CountingLine
    ) -> bool:
        """Check if movement from prev_pos to curr_pos crosses the line."""
        prev_side = self._point_side_of_line(prev_pos, line)
        curr_side = self._point_side_of_line(curr_pos, line)

        # Crossing occurs if signs are different (and neither is zero)
        return (prev_side * curr_side) < 0

    def _get_crossing_direction(
        self,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float]
    ) -> CrossingDirection:
        """Determine crossing direction based on movement vector."""
        movement = np.array([
            curr_pos[0] - prev_pos[0],
            curr_pos[1] - prev_pos[1]
        ])

        # Dot product with IN direction
        dot = np.dot(movement, self.in_direction)

        if dot > 0:
            return CrossingDirection.IN
        elif dot < 0:
            return CrossingDirection.OUT
        return CrossingDirection.UNKNOWN

    def update(
        self,
        tracks: List,  # List[Track] from tracker
        frame_id: int = 0
    ) -> List[CrossingEvent]:
        """
        Update counter with new tracking results.

        Args:
            tracks: List of Track objects from tracker
            frame_id: Current frame ID

        Returns:
            List of new crossing events
        """
        new_events = []

        for track in tracks:
            # Need at least 2 points in trajectory to check crossing
            if len(track.trajectory) < 2:
                continue

            prev_pos = track.trajectory[-2]
            curr_pos = track.trajectory[-1]

            for line in self.lines:
                # Skip if already counted for this line
                if track.track_id in self._crossed_tracks[line.name]:
                    continue

                # Check for line crossing
                if self._check_line_crossing(prev_pos, curr_pos, line):
                    direction = self._get_crossing_direction(prev_pos, curr_pos)

                    # Record crossing
                    self._crossed_tracks[line.name].add(track.track_id)

                    # Update stats
                    stats = self._stats[line.name]
                    if direction == CrossingDirection.IN:
                        stats.count_in += 1
                        stats.by_class_in[track.class_name] = \
                            stats.by_class_in.get(track.class_name, 0) + 1
                    else:
                        stats.count_out += 1
                        stats.by_class_out[track.class_name] = \
                            stats.by_class_out.get(track.class_name, 0) + 1

                    # Create event
                    event = CrossingEvent(
                        track_id=track.track_id,
                        line_name=line.name,
                        direction=direction,
                        class_name=track.class_name,
                        timestamp=datetime.now(),
                        frame_id=frame_id,
                        position=curr_pos
                    )
                    self._events.append(event)
                    new_events.append(event)

                    logger.debug(
                        f"Vehicle {track.track_id} ({track.class_name}) "
                        f"crossed {line.name} going {direction.value}"
                    )

        return new_events

    def get_stats(self, line_name: Optional[str] = None) -> Dict:
        """
        Get counting statistics.

        Args:
            line_name: Specific line name, or None for all lines

        Returns:
            Dictionary with counting statistics
        """
        if line_name:
            return self._stats[line_name].to_dict()

        return {
            name: stats.to_dict()
            for name, stats in self._stats.items()
        }

    def get_total_counts(self) -> Dict[str, int]:
        """Get total IN/OUT counts across all lines."""
        total_in = sum(s.count_in for s in self._stats.values())
        total_out = sum(s.count_out for s in self._stats.values())
        return {
            "in": total_in,
            "out": total_out,
            "total": total_in + total_out
        }

    def get_events(self, limit: Optional[int] = None) -> List[Dict]:
        """Get crossing events as list of dictionaries."""
        events = self._events[-limit:] if limit else self._events
        return [e.to_dict() for e in events]

    def reset(self):
        """Reset all counters and events."""
        for line in self.lines:
            self._crossed_tracks[line.name].clear()
            self._stats[line.name] = CountingStats(line_name=line.name)
        self._events.clear()

    @property
    def counting_lines(self) -> List[CountingLine]:
        """Get all counting lines."""
        return self.lines

    @property
    def total_events(self) -> int:
        """Get total number of crossing events."""
        return len(self._events)
