"""
Visualization module for traffic analytics.
Draws bounding boxes, trajectories, counting lines, and statistics.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import time


# Color palette for tracks (BGR format)
TRACK_COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Pink
    (255, 128, 0),  # Light blue
    (0, 128, 255),  # Orange
    (128, 255, 0),  # Light green
]

# Class-specific colors (BGR)
CLASS_COLORS = {
    "car": (0, 255, 0),        # Green
    "motorcycle": (255, 0, 0),  # Blue
    "bus": (0, 165, 255),       # Orange
    "truck": (255, 0, 255),     # Magenta
}


class Visualizer:
    """
    Visualization handler for traffic analytics.

    Draws all visual elements on video frames.
    """

    def __init__(
        self,
        show_boxes: bool = True,
        show_trajectories: bool = True,
        trajectory_length: int = 30,
        show_ids: bool = True,
        show_labels: bool = True,
        show_confidence: bool = False,
        show_fps: bool = True,
        show_stats: bool = True,
        box_thickness: int = 2,
        font_scale: float = 0.6
    ):
        """
        Initialize visualizer.

        Args:
            show_boxes: Draw bounding boxes
            show_trajectories: Draw track trajectories
            trajectory_length: Max trajectory points to draw
            show_ids: Show track IDs
            show_labels: Show class labels
            show_confidence: Show confidence scores
            show_fps: Show FPS counter
            show_stats: Show counting statistics
            box_thickness: Bounding box line thickness
            font_scale: Font scale for text
        """
        self.show_boxes = show_boxes
        self.show_trajectories = show_trajectories
        self.trajectory_length = trajectory_length
        self.show_ids = show_ids
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.show_fps = show_fps
        self.show_stats = show_stats
        self.box_thickness = box_thickness
        self.font_scale = font_scale

        # FPS calculation
        self._fps_buffer = deque(maxlen=30)
        self._last_time = time.time()

    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get color for a track based on ID."""
        return TRACK_COLORS[track_id % len(TRACK_COLORS)]

    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for a class."""
        return CLASS_COLORS.get(class_name, (255, 255, 255))

    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        current_time = time.time()
        dt = current_time - self._last_time
        self._last_time = current_time

        if dt > 0:
            self._fps_buffer.append(1.0 / dt)

        return np.mean(self._fps_buffer) if self._fps_buffer else 0.0

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List,  # List[Track]
        use_class_colors: bool = True
    ) -> np.ndarray:
        """
        Draw tracked objects on frame.

        Args:
            frame: Input frame
            tracks: List of Track objects
            use_class_colors: Use class-specific colors instead of track colors

        Returns:
            Frame with drawn elements
        """
        output = frame.copy()

        for track in tracks:
            # Get color
            if use_class_colors:
                color = self._get_class_color(track.class_name)
            else:
                color = self._get_track_color(track.track_id)

            # Draw bounding box
            if self.show_boxes:
                x1, y1, x2, y2 = map(int, track.bbox)
                cv2.rectangle(output, (x1, y1), (x2, y2), color, self.box_thickness)

            # Draw trajectory
            if self.show_trajectories and len(track.trajectory) > 1:
                points = np.array(track.trajectory, dtype=np.int32)
                # Draw as polyline with fading effect
                for i in range(1, len(points)):
                    # Fade from transparent to opaque
                    alpha = i / len(points)
                    thickness = max(1, int(self.box_thickness * alpha))
                    pt1 = tuple(points[i - 1])
                    pt2 = tuple(points[i])
                    cv2.line(output, pt1, pt2, color, thickness)

            # Draw label
            if self.show_ids or self.show_labels or self.show_confidence:
                x1, y1 = int(track.bbox[0]), int(track.bbox[1])

                # Build label text
                label_parts = []
                if self.show_ids:
                    label_parts.append(f"ID:{track.track_id}")
                if self.show_labels:
                    label_parts.append(track.class_name)
                if self.show_confidence:
                    label_parts.append(f"{track.confidence:.2f}")

                label = " ".join(label_parts)

                # Draw label background
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
                )
                cv2.rectangle(
                    output,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w + 4, y1),
                    color,
                    -1
                )

                # Draw text
                cv2.putText(
                    output,
                    label,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        return output

    def draw_counting_lines(
        self,
        frame: np.ndarray,
        lines: List,  # List[CountingLine]
        stats: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Draw counting lines on frame.

        Args:
            frame: Input frame
            lines: List of CountingLine objects
            stats: Optional statistics dictionary

        Returns:
            Frame with drawn lines
        """
        output = frame.copy()

        for line in lines:
            # Draw line
            cv2.line(
                output,
                line.start,
                line.end,
                line.color,
                line.thickness + 1
            )

            # Draw line name and counts
            mid_x = (line.start[0] + line.end[0]) // 2
            mid_y = (line.start[1] + line.end[1]) // 2

            # Get stats for this line
            if stats and line.name in stats:
                line_stats = stats[line.name]
                count_text = f"{line.name}: IN={line_stats['count_in']} OUT={line_stats['count_out']}"
            else:
                count_text = line.name

            # Draw text with background
            (text_w, text_h), _ = cv2.getTextSize(
                count_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            cv2.rectangle(
                output,
                (mid_x - text_w // 2 - 5, mid_y - text_h - 10),
                (mid_x + text_w // 2 + 5, mid_y + 5),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                output,
                count_text,
                (mid_x - text_w // 2, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                line.color,
                2,
                cv2.LINE_AA
            )

        return output

    def draw_stats_panel(
        self,
        frame: np.ndarray,
        total_counts: Dict,
        class_counts: Optional[Dict] = None,
        position: str = "top-left"
    ) -> np.ndarray:
        """
        Draw statistics panel on frame.

        Args:
            frame: Input frame
            total_counts: Total IN/OUT counts
            class_counts: Counts by vehicle class
            position: Panel position ("top-left", "top-right", etc.)

        Returns:
            Frame with stats panel
        """
        output = frame.copy()

        # Build stats text lines
        lines = [
            f"Total: {total_counts.get('total', 0)}",
            f"IN: {total_counts.get('in', 0)}  OUT: {total_counts.get('out', 0)}"
        ]

        if class_counts:
            for class_name, count in class_counts.items():
                lines.append(f"  {class_name}: {count}")

        # Calculate panel size
        max_width = 0
        total_height = 10
        for line in lines:
            (text_w, text_h), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            max_width = max(max_width, text_w)
            total_height += text_h + 10

        # Determine position
        margin = 10
        if position == "top-left":
            x, y = margin, margin
        elif position == "top-right":
            x, y = frame.shape[1] - max_width - margin - 20, margin
        elif position == "bottom-left":
            x, y = margin, frame.shape[0] - total_height - margin
        else:  # bottom-right
            x, y = frame.shape[1] - max_width - margin - 20, frame.shape[0] - total_height - margin

        # Draw panel background
        cv2.rectangle(
            output,
            (x, y),
            (x + max_width + 20, y + total_height),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            output,
            (x, y),
            (x + max_width + 20, y + total_height),
            (255, 255, 255),
            1
        )

        # Draw text
        text_y = y + 25
        for line in lines:
            cv2.putText(
                output,
                line,
                (x + 10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            text_y += 25

        return output

    def draw_fps(
        self,
        frame: np.ndarray,
        position: str = "top-right"
    ) -> np.ndarray:
        """
        Draw FPS counter on frame.

        Args:
            frame: Input frame
            position: Text position

        Returns:
            Frame with FPS counter
        """
        output = frame.copy()
        fps = self._calculate_fps()

        fps_text = f"FPS: {fps:.1f}"
        (text_w, text_h), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2
        )

        # Position
        margin = 10
        if position == "top-right":
            x = frame.shape[1] - text_w - margin - 10
            y = margin + text_h + 5
        else:  # top-left
            x = margin
            y = margin + text_h + 5

        # Draw background
        cv2.rectangle(
            output,
            (x - 5, y - text_h - 5),
            (x + text_w + 5, y + 5),
            (0, 0, 0),
            -1
        )

        # Draw text
        color = (0, 255, 0) if fps > 20 else (0, 165, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(
            output,
            fps_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            color,
            2,
            cv2.LINE_AA
        )

        return output

    def render(
        self,
        frame: np.ndarray,
        tracks: List = None,
        counting_lines: List = None,
        counting_stats: Dict = None,
        total_counts: Dict = None
    ) -> np.ndarray:
        """
        Render all visualizations on frame.

        Args:
            frame: Input frame
            tracks: List of Track objects
            counting_lines: List of CountingLine objects
            counting_stats: Statistics by line
            total_counts: Total IN/OUT counts

        Returns:
            Fully rendered frame
        """
        output = frame.copy()

        # Draw counting lines first (background)
        if counting_lines:
            output = self.draw_counting_lines(output, counting_lines, counting_stats)

        # Draw tracks
        if tracks:
            output = self.draw_tracks(output, tracks)

        # Draw stats panel
        if self.show_stats and total_counts:
            output = self.draw_stats_panel(output, total_counts, position="top-left")

        # Draw FPS
        if self.show_fps:
            output = self.draw_fps(output, position="top-right")

        return output


class VideoWriter:
    """Simple video writer wrapper."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v"
    ):
        """
        Initialize video writer.

        Args:
            output_path: Output file path
            fps: Frames per second
            width: Frame width
            height: Frame height
            codec: Video codec (default: mp4v)
        """
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray):
        """Write a frame."""
        self.writer.write(frame)

    def release(self):
        """Release the writer."""
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
