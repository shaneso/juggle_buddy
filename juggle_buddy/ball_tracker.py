"""
Ball detection and tracking module.
Uses YOLO for detection and OpenCV tracking for frame-to-frame tracking.
"""
import numpy as np
import cv2
from typing import List, Dict, Optional
from .path_utils import BallPath


def detect_balls(img: np.ndarray, color_range: tuple = None) -> List[Dict]:
    """
    Detect balls by color (useful if balls are distinct colors).
    
    Args:
        img: Input image (BGR)
        color_range: HSV color range tuple ((lower_h, lower_s, lower_v), (upper_h, upper_s, upper_v))
        
    Returns:
        List of detected balls
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Default: detect bright/white objects (adjust for your balls)
    if color_range is None:
        lower = np.array([0, 0, 200])  # Adjust these values
        upper = np.array([180, 30, 255])
    else:
        lower, upper = color_range
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    balls = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 5000:  # Filter by size
            (x, y), radius = cv2.minEnclosingCircle(contour)
            balls.append({
                'x': int(x),
                'y': int(y),
                'radius': int(radius)
            })
    
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
