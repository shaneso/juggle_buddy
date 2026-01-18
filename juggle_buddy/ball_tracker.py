"""
Ball detection and tracking module.
Uses YOLO for detection and OpenCV tracking for frame-to-frame tracking.
"""
import numpy as np
import cv2
from typing import List, Dict, Optional
from pathlib import Path
from .path_utils import BallPath

# Global YOLO model cache (loaded once, reused across frames)
_yolo_model = None

# Cached calibrated color ranges (loaded once if config exists)
_calibrated_color_ranges = None


def _load_calibrated_color_ranges() -> Optional[dict]:
    """
    Load calibrated color ranges from data/ball_color_config.py if it exists.
    
    Returns:
        Dictionary of color ranges, or None if config doesn't exist
    """
    global _calibrated_color_ranges
    
    if _calibrated_color_ranges is not None:
        return _calibrated_color_ranges
    
    config_path = Path(__file__).parent.parent / "data" / "ball_color_config.py"
    
    if config_path.exists():
        try:
            # Read and execute the config file
            import importlib.util
            spec = importlib.util.spec_from_file_location("ball_color_config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            if hasattr(config_module, 'BALL_COLOR_RANGES'):
                _calibrated_color_ranges = config_module.BALL_COLOR_RANGES
                print(f"✅ Loaded calibrated color ranges from {config_path}")
                return _calibrated_color_ranges
        except Exception as e:
            print(f"⚠️  Failed to load color config: {e}")
            print("   Using default color ranges")
    
    return None


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


def detect_balls(img: np.ndarray, use_yolo: bool = False, confidence: float = 0.25, 
                 model_path: Optional[str] = None, color_ranges: dict = None,
                 use_multi_color: bool = True) -> List[Dict]:
    """
    Detect balls in an image using YOLO (or fallback to color-based detection).
    
    By default, uses multi-color detection (red, blue, green) which is more reliable
    than YOLO for specific ball colors.
    
    Args:
        img: Input image (BGR format)
        use_yolo: If True, use YOLO detection. If False (default), use color-based detection.
        confidence: Confidence threshold for YOLO detections (0.0-1.0)
        model_path: Optional path to custom YOLO model file
        color_ranges: Dict of HSV color ranges for color detection (see _detect_balls_color)
        use_multi_color: If True, detect multiple ball colors simultaneously (default)
        
    Returns:
        List of detected balls, each as a dict with 'x', 'y', 'radius'
    """
    if use_yolo:
        try:
            return _detect_balls_yolo(img, confidence, model_path)
        except Exception as e:
            print(f"⚠️  YOLO detection failed ({e}), falling back to color-based detection")
            return _detect_balls_color(img, color_ranges, use_multi_color)
    else:
        return _detect_balls_color(img, color_ranges, use_multi_color)


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


def _detect_balls_color(img: np.ndarray, color_ranges: dict = None, use_multi_color: bool = True) -> List[Dict]:
    """
    Fallback: Detect balls using color-based detection (HSV color masking).
    Supports detecting multiple colors (red, blue, green) simultaneously.
    
    Args:
        img: Input image (BGR format)
        color_ranges: Dict of color names to HSV ranges, e.g.:
                     {'red': (lower, upper), 'blue': (lower, upper), 'green': (lower, upper)}
        use_multi_color: If True, detect all configured colors. If False, use single default range.
        
    Returns:
        List of detected balls
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Default HSV ranges for red, blue, green balls
    # These are starting values - calibrate using scripts/calibrate_ball_colors.py
    DEFAULT_COLOR_RANGES = {
        'red': [
            (np.array([0, 120, 70]), np.array([10, 255, 255])),  # Red range 1 (wraps around)
            (np.array([170, 120, 70]), np.array([180, 255, 255]))  # Red range 2
        ],
        'blue': [(np.array([100, 150, 0]), np.array([124, 255, 255]))],
        'green': [(np.array([35, 50, 50]), np.array([85, 255, 255]))]
    }
    
    # Use provided ranges or defaults (calibration disabled - using default ranges)
    if color_ranges is None:
        # Use default color ranges (calibrated ranges loading disabled)
        color_ranges = DEFAULT_COLOR_RANGES
    
    # Create combined mask for all colors
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    if use_multi_color:
        # Detect all configured colors
        for color_name, ranges in color_ranges.items():
            if isinstance(ranges, list):
                # Multiple ranges (e.g., red has two ranges)
                for lower, upper in ranges:
                    mask = cv2.inRange(hsv, lower, upper)
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
            else:
                # Single range
                lower, upper = ranges
                mask = cv2.inRange(hsv, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
    else:
        # Fallback: use first range or default bright/white
        if color_ranges and len(color_ranges) > 0:
            first_ranges = list(color_ranges.values())[0]
            if isinstance(first_ranges, list):
                lower, upper = first_ranges[0]
            else:
                lower, upper = first_ranges
        else:
            # Default: detect bright/white objects
            lower = np.array([0, 0, 200])
            upper = np.array([180, 30, 255])
        combined_mask = cv2.inRange(hsv, lower, upper)
    
    # Apply morphological operations to close gaps and fill holes
    # This is especially important for tape-wrapped balls with gaps between strips
    # Use moderate kernel size - too large connects unrelated noise, too small misses tape gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Moderate size
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))   # Remove tiny noise
    # Open first to remove small noise, then close to connect tape pieces
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)    # Remove tiny noise first
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)  # Then close gaps between tape
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    balls = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter 1: Area range - strict minimum to reject small noise
        # Increased minimum from 100 to 300 to filter out tiny artifacts
        # Typical range: 300-4000 pixels for balls at reasonable distances
        if not (300 <= area <= 4000):
            continue
        
        # Get enclosing circle and bounding box
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter 2: Radius range - strict minimum to reject tiny noise
        # Increased minimum from 8 to 15 pixels
        if not (15 <= radius <= 50):
            continue
        
        # Filter 3: Circularity - balance between tape-wrapped balls and noise
        # Circularity = 4π*area/perimeter² (1.0 = perfect circle)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            # Increased threshold from 0.25 to 0.35 to reject very irregular noise
            if circularity < 0.35:
                continue
        
        # Filter 4: Area-to-circle ratio - stricter minimum for better signal
        # Helps reject sparse/noisy detections
        circle_area = np.pi * radius * radius
        if circle_area > 0:
            fill_ratio = area / circle_area
            # Increased minimum from 0.3 to 0.45 to reject sparse shapes
            # Upper bound 1.3 (was 1.5) to be stricter
            if not (0.45 <= fill_ratio <= 1.3):
                continue
        
        # Filter 5: Bounding box aspect ratio - stricter to reject elongated noise
        if w > 0 and h > 0:
            aspect_ratio = max(w, h) / min(w, h)
            # Reduced from 2.0 to 1.6 to reject elongated artifacts
            if aspect_ratio > 1.6:
                continue
        
        # Filter 6: Solidity - how convex is the shape (ratio of contour area to convex hull area)
        # Helps reject very irregular shapes that are more likely to be noise
        # Solidity = contour_area / convex_hull_area (1.0 = perfectly convex)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            # Reject shapes that are too concave/irregular (solidity < 0.7)
            # This helps filter out noise while allowing some irregularities from tape
            if solidity < 0.7:
                continue
        
        balls.append({
            'x': int(cx),
            'y': int(cy),
            'radius': radius
        })
    
    # Filter 7: Remove detections that are too close together (likely duplicates)
    # Keep only the largest detection within a threshold distance
    if len(balls) > 1:
        filtered_balls = []
        balls_sorted = sorted(balls, key=lambda b: b['radius'], reverse=True)  # Sort by radius (largest first)
        
        for ball in balls_sorted:
            too_close = False
            for existing in filtered_balls:
                dist = np.sqrt((ball['x'] - existing['x'])**2 + (ball['y'] - existing['y'])**2)
                # If within 30 pixels of another detection, skip this one (keep the larger one)
                if dist < 30:
                    too_close = True
                    break
            if not too_close:
                filtered_balls.append(ball)
        
        balls = filtered_balls
    
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
