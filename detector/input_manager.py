"""
Input Manager Module - Multi-source Video Input Handling

This module provides the InputManager class for handling various input sources
including webcam, YouTube videos, and local video files.

Author: [Your Name]
License: MIT
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Generator, Union

import cv2
import numpy as np

# Try to import pytube
try:
    import pytube
    from pytube.exceptions import PytubeError
    PYTUBE_AVAILABLE = True
except ImportError:
    PYTUBE_AVAILABLE = False
    PytubeError = Exception  # Fallback

# Try to import yt-dlp as fallback
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False


class InputManager:
    """Manages multiple input sources for video processing.

    This class handles switching between webcam, YouTube videos, and local files
    with proper error handling and resource management.

    Attributes:
        current_source: Current video capture object
        source_type: Type of current input source
        logger: Logger instance for debugging

    Example:
        >>> input_manager = InputManager()
        >>> input_manager.set_webcam_source(0)
        >>> for frame in input_manager.get_frames():
        ...     # Process frame
        ...     pass
    """

    def __init__(self) -> None:
        """Initialize input manager."""
        self.logger = logging.getLogger(__name__)
        self.current_source: Optional[cv2.VideoCapture] = None
        self.source_type: str = "none"
        self.temp_dir = Path("videos")
        self.temp_dir.mkdir(exist_ok=True)

        # YouTube video handling
        self.youtube_streams = {}

    def set_webcam_source(self, camera_id: int = 0) -> bool:
        """Set webcam as input source.

        Args:
            camera_id: Webcam device ID (usually 0 for default)

        Returns:
            True if webcam successfully opened, False otherwise
        """
        try:
            self._cleanup_current_source()

            self.logger.info(f"Setting webcam source: camera_id={camera_id}")
            cap = cv2.VideoCapture(camera_id)

            if not cap.isOpened():
                self.logger.error(f"Failed to open webcam {camera_id}")
                return False

            # Set webcam properties for 720p HD
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # Verify settings
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)

            self.logger.info(f"Webcam configured: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")

            self.current_source = cap
            self.source_type = "webcam"
            return True

        except Exception as e:
            self.logger.error(f"Error setting webcam source: {str(e)}")
            return False

    def set_video_file_source(self, file_path: Union[str, Path]) -> bool:
        """Set local video file as input source.

        Args:
            file_path: Path to video file

        Returns:
            True if video file successfully opened, False otherwise
        """
        try:
            self._cleanup_current_source()

            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.error(f"Video file not found: {file_path}")
                return False

            self.logger.info(f"Setting video file source: {file_path}")
            cap = cv2.VideoCapture(str(file_path))

            if not cap.isOpened():
                self.logger.error(f"Failed to open video file: {file_path}")
                return False

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.logger.info(f"Video loaded: {width}x{height} @ {fps:.1f}fps, {frame_count} frames")

            self.current_source = cap
            self.source_type = "video_file"
            return True

        except Exception as e:
            self.logger.error(f"Error setting video file source: {str(e)}")
            return False

    def set_youtube_source(self, youtube_url: str, quality: str = "720p") -> bool:
        """Set YouTube video as input source.

        Args:
            youtube_url: YouTube video URL
            quality: Video quality preference ("360p", "720p", "1080p")

        Returns:
            True if YouTube video successfully loaded, False otherwise
        """
        try:
            self._cleanup_current_source()

            self.logger.info(f"Loading YouTube video: {youtube_url}")

            # Try pytube first if available
            if PYTUBE_AVAILABLE and self._try_pytube_download(youtube_url, quality):
                return True

            # Fallback to yt-dlp if pytube fails or not available
            if YT_DLP_AVAILABLE and self._try_yt_dlp_download(youtube_url, quality):
                return True

            self.logger.error("No YouTube downloader available (install pytube or yt-dlp)")
            return False

        except Exception as e:
            self.logger.error(f"Error setting YouTube source: {str(e)}")
            return False

    def _try_pytube_download(self, youtube_url: str, quality: str) -> bool:
        """Try to download video using pytube.

        Args:
            youtube_url: YouTube video URL
            quality: Video quality preference

        Returns:
            True if successful, False otherwise
        """
        try:
            yt = pytube.YouTube(youtube_url)

            # Filter streams by resolution
            streams = yt.streams.filter(progressive=True, file_extension='mp4')

            # Try to get requested quality
            if quality == "1080p" and streams.filter(resolution="1080p"):
                stream = streams.filter(resolution="1080p").first()
            elif quality == "720p" and streams.filter(resolution="720p"):
                stream = streams.filter(resolution="720p").first()
            elif quality == "360p" and streams.filter(resolution="360p"):
                stream = streams.filter(resolution="360p").first()
            else:
                # Fallback to highest available quality
                stream = streams.order_by('resolution').desc().first()

            if not stream:
                return False

            # Download video to temporary directory
            self.logger.info(f"Downloading {stream.resolution} video with pytube...")
            temp_path = self.temp_dir / f"youtube_{yt.video_id}.mp4"

            if not temp_path.exists():
                stream.download(output_path=str(self.temp_dir), filename=f"youtube_{yt.video_id}.mp4")

            # Open downloaded video
            cap = cv2.VideoCapture(str(temp_path))
            if not cap.isOpened():
                return False

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            self.logger.info(f"YouTube video loaded with pytube: {width}x{height} @ {fps:.1f}fps")
            self.logger.info(f"Video title: {yt.title}")

            self.current_source = cap
            self.source_type = "youtube"
            self.youtube_streams[youtube_url] = temp_path

            return True

        except PytubeError as e:
            self.logger.warning(f"Pytube failed: {str(e)}")
            return False

    def _try_yt_dlp_download(self, youtube_url: str, quality: str) -> bool:
        """Try to download video using yt-dlp.

        Args:
            youtube_url: YouTube video URL
            quality: Video quality preference

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Trying yt-dlp for YouTube video...")

            # Configure yt-dlp
            ydl_opts = {
                'format': f'best[height<={quality.replace("p", "")}]',
                'outtmpl': str(self.temp_dir / 'youtube_%(id)s.%(ext)s'),
                'quiet': False,  # Show progress
                'no_warnings': False,
                'noprogress': False,  # Show download progress
            }

            # Extract video info and download
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                video_id = info['id']
                title = info.get('title', 'Unknown')
                temp_path = self.temp_dir / f"youtube_{video_id}.mp4"

                # Check if download succeeded
                if not temp_path.exists():
                    # Try with .webm extension
                    temp_path = self.temp_dir / f"youtube_{video_id}.webm"
                    if not temp_path.exists():
                        # Try with .mkv extension
                        temp_path = self.temp_dir / f"youtube_{video_id}.mkv"

            # Open downloaded video
            cap = cv2.VideoCapture(str(temp_path))
            if not cap.isOpened():
                self.logger.error("Failed to open downloaded YouTube video")
                return False

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            self.logger.info(f"YouTube video loaded with yt-dlp: {width}x{height} @ {fps:.1f}fps")
            self.logger.info(f"Video title: {title}")

            self.current_source = cap
            self.source_type = "youtube"
            self.youtube_streams[youtube_url] = temp_path

            return True

        except Exception as e:
            self.logger.error(f"yt-dlp failed: {str(e)}")
            return False

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get a single frame from current source.

        Returns:
            Tuple of (success, frame) where frame is numpy array or None
        """
        if self.current_source is None:
            return False, None

        try:
            ret, frame = self.current_source.read()
            if not ret:
                self.logger.warning("Failed to read frame from current source")
                return False, None

            return True, frame

        except Exception as e:
            self.logger.error(f"Error reading frame: {str(e)}")
            return False, None

    def get_frames(self) -> Generator[Tuple[bool, np.ndarray], None, None]:
        """Generator that yields frames from current source.

        Yields:
            Tuple of (success, frame) for each frame
        """
        while True:
            success, frame = self.get_frame()
            if not success or frame is None:
                break
            yield success, frame

    def get_source_info(self) -> dict:
        """Get information about current input source.

        Returns:
            Dictionary containing source information
        """
        info = {
            'source_type': self.source_type,
            'is_open': self.current_source is not None and self.current_source.isOpened()
        }

        if self.current_source is not None and info['is_open']:
            info.update({
                'width': int(self.current_source.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.current_source.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.current_source.get(cv2.CAP_PROP_FPS),
                'frame_count': int(self.current_source.get(cv2.CAP_PROP_FRAME_COUNT)) if self.source_type != "webcam" else -1
            })

        return info

    def reset_source(self) -> bool:
        """Reset current source to beginning.

        Returns:
            True if reset successful, False otherwise
        """
        if self.current_source is None:
            return False

        try:
            if self.source_type != "webcam":
                # Only non-webcam sources can be reset
                self.current_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.logger.info(f"Reset {self.source_type} source to beginning")
                return True
            else:
                self.logger.warning("Cannot reset webcam source")
                return False

        except Exception as e:
            self.logger.error(f"Error resetting source: {str(e)}")
            return False

    def _cleanup_current_source(self) -> None:
        """Clean up current video source."""
        if self.current_source is not None:
            try:
                self.current_source.release()
                self.logger.debug(f"Released {self.source_type} source")
            except Exception as e:
                self.logger.error(f"Error releasing source: {str(e)}")
            finally:
                self.current_source = None
                self.source_type = "none"

    def cleanup_temp_files(self) -> None:
        """Clean up temporary YouTube video files."""
        try:
            for file_path in self.temp_dir.glob("youtube_*.mp4"):
                file_path.unlink()
                self.logger.debug(f"Deleted temporary file: {file_path}")
        except Exception as e:
            self.logger.error(f"Error cleaning temp files: {str(e)}")

    def __del__(self):
        """Cleanup when input manager is destroyed."""
        self._cleanup_current_source()
        self.cleanup_temp_files()


# Test function for development
def test_input_manager():
    """Test the input manager with different sources."""
    import sys

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    input_manager = InputManager()

    try:
        # Test webcam
        print("Testing webcam...")
        if input_manager.set_webcam_source(0):
            info = input_manager.get_source_info()
            print(f"Webcam info: {info}")

            # Read a few frames
            for i in range(5):
                success, frame = input_manager.get_frame()
                if success:
                    print(f"Frame {i+1}: {frame.shape}")
                else:
                    print(f"Failed to read frame {i+1}")
        else:
            print("Failed to open webcam")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {str(e)}")
    finally:
        input_manager._cleanup_current_source()


if __name__ == "__main__":
    test_input_manager()