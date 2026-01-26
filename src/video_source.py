"""
Video source handler for various input types.
Supports: local files, RTSP streams, HTTP streams, YouTube.
"""

import cv2
import logging
from typing import Generator, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SourceType(Enum):
    FILE = "file"
    RTSP = "rtsp"
    HTTP = "http"
    HLS = "hls"
    YOUTUBE = "youtube"


@dataclass
class FrameInfo:
    """Container for frame data and metadata."""
    frame: any  # numpy array
    frame_id: int
    timestamp: float  # milliseconds
    fps: float


class VideoSource:
    """
    Universal video source handler.

    Supports multiple input types with automatic reconnection
    for streaming sources.
    """

    def __init__(
        self,
        source: str,
        source_type: str = "file",
        frame_skip: int = 1,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0
    ):
        """
        Initialize video source.

        Args:
            source: Path to video file or stream URL
            source_type: One of "file", "rtsp", "http", "youtube"
            frame_skip: Process every N-th frame (1 = all frames)
            reconnect_attempts: Number of reconnection attempts for streams
            reconnect_delay: Delay between reconnection attempts (seconds)
        """
        self.source = source
        self.source_type = SourceType(source_type)
        self.frame_skip = max(1, frame_skip)
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self._fps = 30.0
        self._width = 0
        self._height = 0

    def _get_stream_url(self) -> str:
        """Get the actual URL for the video source."""
        if self.source_type == SourceType.YOUTUBE:
            try:
                from vidgear.gears import CamGear
                # Use vidgear to extract YouTube stream URL
                stream = CamGear(
                    source=self.source,
                    stream_mode=True,
                    logging=False
                )
                stream.start()
                return stream
            except ImportError:
                logger.warning("vidgear not installed. Install with: pip install vidgear")
                raise
        return self.source

    def open(self) -> bool:
        """
        Open the video source.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.source_type == SourceType.YOUTUBE:
                self._stream = self._get_stream_url()
                # Get properties from first frame
                frame = self._stream.read()
                if frame is not None:
                    self._height, self._width = frame.shape[:2]
                    self._fps = 30.0  # YouTube streams typically 30fps
                    return True
                return False

            # For file/RTSP/HTTP/HLS sources
            url = self.source

            # Set transport options for better reliability
            if self.source_type == SourceType.RTSP:
                import os
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

            # For HLS streams, use FFmpeg backend with buffer settings
            if self.source_type == SourceType.HLS:
                self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                # Set buffer size for smoother streaming
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            else:
                self.cap = cv2.VideoCapture(url)

            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {url}")
                return False

            self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(
                f"Opened video source: {self._width}x{self._height} @ {self._fps:.1f} FPS"
            )
            return True

        except Exception as e:
            logger.error(f"Error opening video source: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[FrameInfo]]:
        """
        Read next frame from source.

        Returns:
            Tuple of (success, FrameInfo or None)
        """
        if self.source_type == SourceType.YOUTUBE:
            frame = self._stream.read()
            if frame is None:
                return False, None
            self.frame_count += 1
            return True, FrameInfo(
                frame=frame,
                frame_id=self.frame_count,
                timestamp=self.frame_count * (1000.0 / self._fps),
                fps=self._fps
            )

        if self.cap is None:
            return False, None

        # Skip frames if needed
        for _ in range(self.frame_skip - 1):
            self.cap.grab()
            self.frame_count += 1

        ret, frame = self.cap.read()

        if not ret:
            # Try to reconnect for streaming sources
            if self.source_type in (SourceType.RTSP, SourceType.HTTP, SourceType.HLS):
                return self._try_reconnect()
            return False, None

        self.frame_count += 1
        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)

        return True, FrameInfo(
            frame=frame,
            frame_id=self.frame_count,
            timestamp=timestamp,
            fps=self._fps
        )

    def _try_reconnect(self) -> Tuple[bool, Optional[FrameInfo]]:
        """Attempt to reconnect to streaming source."""
        import time

        for attempt in range(self.reconnect_attempts):
            logger.warning(
                f"Connection lost. Reconnecting... ({attempt + 1}/{self.reconnect_attempts})"
            )
            time.sleep(self.reconnect_delay)

            if self.cap is not None:
                self.cap.release()

            if self.open():
                return self.read()

        logger.error("Failed to reconnect to stream")
        return False, None

    def frames(self) -> Generator[FrameInfo, None, None]:
        """
        Generator that yields frames from the source.

        Yields:
            FrameInfo objects for each frame
        """
        if not self.open():
            return

        while True:
            success, frame_info = self.read()
            if not success:
                break
            yield frame_info

        self.release()

    def release(self):
        """Release the video source."""
        if self.source_type == SourceType.YOUTUBE:
            if hasattr(self, '_stream'):
                self._stream.stop()
        elif self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def total_frames(self) -> int:
        """Total frames (only available for file sources)."""
        if self.cap is not None and self.source_type == SourceType.FILE:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
