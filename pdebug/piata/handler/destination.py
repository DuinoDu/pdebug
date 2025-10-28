"""output handler"""
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..registry import OUTPUT_REGISTRY

try:
    import moviepy
    from moviepy.editor import ImageClip, concatenate_videoclips

    MOVIEPY_INSTALLED = True
except ImportError:
    MOVIEPY_INSTALLED = False

__all__ = ["ImgdirWriter", "VideoWriter", "FFmpegVideoWriter"]


@OUTPUT_REGISTRY.register(name="imgdir")
class ImgdirWriter:
    def __init__(
        self,
        images: List[np.ndarray],
        **kwargs,
    ):
        self._images = images
        self._ext = kwargs.get("ext", ".png")

    def save(self, savedir: str):
        os.makedirs(savedir, exist_ok=True)
        # TODO: Fix savename
        for idx, image in enumerate(self._images):
            savename = os.path.join(savedir, f"{idx}{self._ext}")
            cv2.imwrite(savename, image)


@OUTPUT_REGISTRY.register(name="video")
class VideoWriter:

    """Write image list to video using moviepy."""

    def __init__(
        self,
        images: List[np.ndarray],
        **kwargs,
    ):
        self._images = images
        self._write_kwargs = kwargs
        if not MOVIEPY_INSTALLED:
            raise RuntimeError(
                "moviepy is required, please install using pip."
            )

    def save(self, savename: str):
        fps = self._write_kwargs.get("fps", 15)
        duration = 1.0 / fps
        clips = [ImageClip(m).set_duration(duration) for m in self._images]
        for c in clips:
            c.fps = fps
        concat_clip = concatenate_videoclips(clips, method="compose")
        video_ext = os.path.splitext(savename)[1]
        assert video_ext in (".mp4", ".avi", "ogv", ".webm")
        concat_clip.write_videofile(savename, **self._write_kwargs)
        concat_clip.close()


@OUTPUT_REGISTRY.register(name="video_ffmpeg")
class FFMPEGVideoWriter:
    """
    A video writer that uses ffmpeg to write OpenCV images to MP4 files.

    This class provides an efficient way to write video frames using ffmpeg,
    which often provides better compression and quality compared to OpenCV's
    built-in VideoWriter.

    Examples:
        >>> with FFMPEGVideoWriter(output_path, fps=fps, **kwargs) as writer:
        >>>     writer.write_frames(frames)
    """

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        resolution: Optional[Tuple[int, int]] = None,
        codec: str = "libx264",
        crf: int = 23,
        preset: str = "medium",
    ):
        """
        Initialize the FFMPEG video writer.

        Args:
            output_path: Path to the output MP4 file
            fps: Frames per second
            resolution: (width, height) tuple. If None, inferred from first frame
            codec: FFmpeg codec to use (default: libx264)
            crf: Constant Rate Factor for quality (lower = higher quality, 0-51)
            preset: Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.resolution = resolution
        self.codec = codec
        self.crf = crf
        self.preset = preset

        self.process = None
        self.frame_count = 0
        self._is_open = False

    def _start_ffmpeg_process(self, frame_shape: Tuple[int, int, int]):
        """Start the ffmpeg process with appropriate parameters."""
        if self.resolution is None:
            height, width = frame_shape[:2]
        else:
            width, height = self.resolution

        # Ensure output directory exists
        self.output_path.parent.mkdir(exist_ok=True)

        command = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(self.fps),
            "-i",
            "-",  # Read from stdin
            "-c:v",
            self.codec,
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",  # Ensure compatibility
            self.output_path,
        ]

        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            self._is_open = True
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg and ensure it's in your PATH. "
                "Installation instructions: https://ffmpeg.org/download.html"
            )

    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the video.

        Args:
            frame: OpenCV image (numpy array in BGR format)
        """
        if not self._is_open:
            self._start_ffmpeg_process(frame.shape)

        # Validate frame dimensions
        if self.resolution is not None:
            expected_width, expected_height = self.resolution
            if (
                frame.shape[1] != expected_width
                or frame.shape[0] != expected_height
            ):
                frame = cv2.resize(frame, (expected_width, expected_height))

        try:
            self.process.stdin.write(frame.tobytes())
            self.frame_count += 1
        except BrokenPipeError:
            # Read error output for debugging
            error_output = self.process.stderr.read().decode()
            raise RuntimeError(f"FFmpeg process failed: {error_output}")

    def write_frames(self, frames: list):
        """
        Write multiple frames to the video.

        Args:
            frames: List of OpenCV images
        """
        for frame in frames:
            self.write_frame(frame)

    def close(self):
        """Close the video writer and finalize the video file."""
        if self._is_open and self.process is not None:
            try:
                self.process.stdin.close()
                self.process.wait()

                # Check if process completed successfully
                if self.process.returncode != 0:
                    error_output = self.process.stderr.read().decode()
                    raise RuntimeError(
                        f"FFmpeg process failed with return code {self.process.returncode}: {error_output}"
                    )

            except Exception as e:
                # Ensure process is terminated
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait()
                raise e
            finally:
                self._is_open = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @property
    def is_open(self) -> bool:
        """Check if the writer is currently open."""
        return self._is_open

    def get_info(self) -> dict:
        """Get information about the writer state."""
        return {
            "output_path": str(self.output_path),
            "fps": self.fps,
            "resolution": self.resolution,
            "codec": self.codec,
            "crf": self.crf,
            "preset": self.preset,
            "frame_count": self.frame_count,
            "is_open": self._is_open,
        }

    def save(self, output_name=None):
        self.close()
