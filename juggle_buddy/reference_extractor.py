"""
Reference path extraction module.
Extracts and normalizes ideal ball paths from reference video.
"""
import numpy as np
import cv2
from typing import List, Optional
from pathlib import Path
from .ball_tracker import BallTracker
from .path_utils import normalize_path


class ReferenceExtractor:
    """Extracts reference paths from video or tracker."""
    
    def __init__(self, num_balls: int = 3):
        """
        Initialize reference extractor.
        
        Args:
            num_balls: Number of balls in the pattern
        """
        self.num_balls = num_balls
        self.reference_paths: Optional[List[np.ndarray]] = None
    
    def extract_from_video(self, video_path: Optional[str] = None) -> Optional[List[np.ndarray]]:
        """
        Extract reference paths from a video file.
        If no path provided, uses default reference video (videos/cascade.mp4).
        
        Args:
            video_path: Path to reference video file (optional)
            
        Returns:
            List of normalized reference paths, one per ball
        """
        # Use default reference video if not provided
        if video_path is None:
            project_root = Path(__file__).parent.parent
            video_path = str(project_root / "videos" / "cascade.mp4")
        
        try:
            # Create tracker and process video
            tracker = BallTracker(num_balls=self.num_balls)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Warning: Could not open video file: {video_path}")
                return None
            
            print(f"Processing reference video: {video_path}")
            frame_num = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                tracker.process_frame(frame, frame_num)
                frame_num += 1
                
                # Progress indicator
                if frame_num % 30 == 0:
                    print(f"  Processed {frame_num} frames...")
            
            cap.release()
            print(f"Finished processing {frame_num} frames")
            
            # Extract paths
            paths = self.extract_from_tracker(tracker)
            
            # Store reference paths
            self.reference_paths = paths
            
            return paths
            
        except Exception as e:
            print(f"Error extracting reference paths: {e}")
            return None
    
    def extract_from_tracker(self, tracker: BallTracker) -> List[np.ndarray]:
        """
        Extract paths from a BallTracker instance.
        
        Args:
            tracker: BallTracker with processed frames
            
        Returns:
            List of path arrays, one per ball
        """
        paths = tracker.get_paths()
        
        # Convert BallPath objects to numpy arrays
        path_arrays = []
        for path in paths:
            if len(path) > 0:
                path_arrays.append(path.positions)
            else:
                # Empty path
                path_arrays.append(np.array([]).reshape(0, 2))
        
        # Ensure we have the right number of paths
        while len(path_arrays) < self.num_balls:
            path_arrays.append(np.array([]).reshape(0, 2))
        
        return path_arrays[:self.num_balls]


def normalize_reference(paths: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize a set of reference paths.
    Centers all paths around a common origin.
    
    Args:
        paths: List of path arrays
        
    Returns:
        List of normalized path arrays
    """
    if not paths:
        return []
    
    # Find the overall center across all paths
    all_positions = []
    for path in paths:
        if len(path) > 0:
            all_positions.append(path)
    
    if not all_positions:
        return paths
    
    # Calculate global center
    all_coords = np.vstack(all_positions)
    global_center = all_coords.mean(axis=0)
    
    # Normalize each path
    normalized = []
    for path in paths:
        if len(path) > 0:
            normalized.append(path - global_center)
        else:
            normalized.append(path)
    
    return normalized
