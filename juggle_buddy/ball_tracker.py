"""
Ball detection and tracking module.
Uses YOLO for detection and OpenCV tracking for frame-to-frame tracking.
"""
import numpy as np
import cv2
from typing import List, Dict, Optional
from .path_utils import BallPath

# Global YOLO model cache (loaded once, reused across frames)
_yolo_model = None


def _get_yolo_model(model_path: Optional[str] = None):
    """Get or create YOLO model (cached globally for performance)."""
    global _yolo_model
    
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            
            if model_path is None:
                # Use pre-trained COCO model (detects "sports ball" - class 32)
                # First run will download the model automatically
                model_path = 'yolov8n.pt'
            
            _yolo_model = YOLO(model_path)
            print(f"✅ YOLO model loaded: {model_path}")
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    return _yolo_model


def detect_balls(img: np.ndarray, use_yolo: bool = True, confidence: float = 0.25, model_path: Optional[str] = None) -> List[Dict]:
    """
    Detect balls in an image using YOLO (or fallback to color-based detection).
    
    Args:
        img: Input image (BGR format)
        use_yolo: If True, use YOLO detection (default). If False, use color-based fallback.
        confidence: Confidence threshold for YOLO detections (0.0-1.0)
        model_path: Optional path to custom YOLO model file
        
    Returns:
        List of detected balls, each as a dict with 'x', 'y', 'radius'
    """
    if use_yolo:
        try:
            return _detect_balls_yolo(img, confidence, model_path)
        except Exception as e:
            print(f"⚠️  YOLO detection failed ({e}), falling back to color-based detection")
            return _detect_balls_color(img)
    else:
        return _detect_balls_color(img)


def _detect_balls_yolo(img: np.ndarray, confidence: float = 0.25, model_path: Optional[str] = None) -> List[Dict]:
    """
    Detect balls using YOLO v8 (detects "sports ball" from COCO dataset).
    
    Args:
        img: Input image (BGR format)
        confidence: Confidence threshold (0.0-1.0)
        model_path: Optional path to custom YOLO model
        
    Returns:
        List of detected balls
    """
    model = _get_yolo_model(model_path)
    
    # Run YOLO detection (class 32 = sports ball in COCO dataset)
    results = model(img, classes=[32], conf=confidence, verbose=False)
    
    balls = []
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate center and radius
                x = int((x1 + x2) / 2)
                y = int((y1 + y2) / 2)
                width = x2 - x1
                height = y2 - y1
                radius = int(max(width, height) / 2)  # Use larger dimension for radius
                
                balls.append({
                    'x': x,
                    'y': y,
                    'radius': radius,
                    'confidence': float(box.conf[0].cpu().numpy())  # Store confidence
                })
    
    return balls


def _detect_balls_color(img: np.ndarray, color_range: tuple = None) -> List[Dict]:
    """
    Fallback: Detect balls using color-based detection (HSV color masking).
    
    Args:
        img: Input image (BGR format)
        color_range: HSV color range tuple ((lower_h, lower_s, lower_v), (upper_h, upper_s, upper_v))
        
    Returns:
        List of detected balls
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Default: detect bright/white objects (adjust these values for your balls)
    if color_range is None:
        lower = np.array([0, 0, 200])  # Lower HSV: any hue, low saturation, high value (bright)
        upper = np.array([180, 30, 255])  # Upper HSV: any hue, low saturation, max value
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
