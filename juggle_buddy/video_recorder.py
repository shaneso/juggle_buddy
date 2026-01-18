"""
Video recording and playback utilities.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class VideoRecorder:
    """Handles video recording from webcam."""
    
    def __init__(self, output_path: str = "output.mp4", fps: int = 30):
        """
        Initialize video recorder.
        
        Args:
            output_path: Path to save recorded video
            fps: Frames per second for recording
        """
        self.output_path = output_path
        self.fps = fps
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
    
    def start_recording(self, frame_width: int, frame_height: int):
        """
        Start recording video.
        
        Args:
            frame_width: Width of frames
            frame_height: Height of frames
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (frame_width, frame_height)
        )
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the video.
        
        Args:
            frame: Frame to write (BGR format)
        """
        if self.writer is not None:
            self.writer.write(frame)
    
    def stop_recording(self):
        """Stop recording and release resources."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def load_video(video_path: str) -> Tuple[Optional[cv2.VideoCapture], int, int, int]:
    """
    Load a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (VideoCapture object, fps, width, height)
        Returns (None, 0, 0, 0) if video cannot be loaded
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, 0, 0, 0
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return cap, fps, width, height


def get_reference_video_path() -> str:
    """
    Get the path to the reference cascade video.
    
    Returns:
        Path to videos/cascade.mp4
    """
    project_root = Path(__file__).parent.parent
    video_path = project_root / "videos" / "cascade.mp4"
    return str(video_path)
