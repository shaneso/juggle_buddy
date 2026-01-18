"""
Ball detection and tracking module.
Uses YOLO for detection and OpenCV tracking for frame-to-frame tracking.
"""
import numpy as np
import cv2
from typing import List, Dict, Optional
from .path_utils import BallPath


def detect_balls(img: np.ndarray) -> List[Dict]:
    """
    Detect balls in an image.
    
    Args:
        img: Input image (BGR format)
        
    Returns:
        List of detected balls, each as a dict with 'x', 'y', 'radius' or similar
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use HoughCircles for initial detection (simple approach)
    # In production, this would use YOLO
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )
    
    balls = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            balls.append({'x': int(x), 'y': int(y), 'radius': int(r)})
    
    return balls


def track_balls_frame(img: np.ndarray, previous_detections: List[Dict]) -> List[Dict]:
    """
    Track balls from previous frame to current frame.
    
    Args:
        img: Current frame
        previous_detections: Ball detections from previous frame
        
    Returns:
        Updated ball detections with tracking IDs
    """
    # Simple implementation - in production would use OpenCV trackers
    current_detections = detect_balls(img)
    
    # Match detections (simplified - would use Hungarian algorithm in production)
    tracked = []
    for det in current_detections:
        tracked.append(det)
    
    return tracked


class BallTracker:
    """Tracks multiple balls across video frames."""
    
    def __init__(self, num_balls: int = 3):
        """
        Initialize ball tracker.
        
        Args:
            num_balls: Expected number of balls to track
        """
        self.num_balls = num_balls
        self.tracked_balls: List[BallPath] = []
        self.frame_count = 0
        self.trackers = []  # OpenCV trackers (to be initialized)
    
    def process_frame(self, img: np.ndarray, frame_number: int) -> Dict:
        """
        Process a single frame and update ball tracks.
        
        Args:
            img: Current frame image
            frame_number: Frame number in sequence
            
        Returns:
            Dict with 'balls' (list of detections) and 'frame_number'
        """
        self.frame_count = frame_number
        
        # Detect balls
        detections = detect_balls(img)
        
        # Update tracking
        if len(self.tracked_balls) == 0:
            # Initialize paths
            for i, det in enumerate(detections[:self.num_balls]):
                pos = np.array([[det['x'], det['y']]])
                path = BallPath(ball_id=i, positions=pos)
                self.tracked_balls.append(path)
        else:
            # Update existing paths
            for i, det in enumerate(detections[:min(len(self.tracked_balls), len(detections))]):
                if i < len(self.tracked_balls):
                    pos = np.array([[det['x'], det['y']]])
                    self.tracked_balls[i].positions = np.vstack([
                        self.tracked_balls[i].positions,
                        pos
                    ])
        
        return {
            'balls': detections[:self.num_balls],
            'frame_number': frame_number
        }
    
    def get_paths(self) -> List[BallPath]:
        """
        Get all tracked ball paths.
        
        Returns:
            List of BallPath objects
        """
        return self.tracked_balls
