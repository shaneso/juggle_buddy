"""
Body scaling module.
Detects body keypoints and scales reference paths to match live performer.
"""
import numpy as np
import cv2
from typing import Dict, List, Optional

# Global YOLO pose model (initialized once)
_yolo_pose_model = None


def _get_yolo_pose_model(model_path: Optional[str] = None):
    """Get or create YOLO pose model (cached globally for performance)."""
    global _yolo_pose_model
    
    if _yolo_pose_model is None:
        try:
            from ultralytics import YOLO
            
            if model_path is None:
                # Use YOLOv8 nano pose model (lightweight and fast)
                # First run will download the model automatically (~6MB)
                model_path = 'yolov8n-pose.pt'
            
            _yolo_pose_model = YOLO(model_path)
            print(f"✅ YOLO pose model loaded: {model_path}")
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO pose model: {e}")
    
    return _yolo_pose_model


def detect_body_keypoints(img: np.ndarray, use_yolo: bool = True, confidence: float = 0.25, model_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Detect body keypoints (shoulders, hips) using YOLOv8 pose estimation.
    
    Args:
        img: Input image (BGR format)
        use_yolo: If True (default), use YOLO pose. If False, use fallback.
        confidence: Confidence threshold for YOLO detections (0.0-1.0)
        model_path: Optional path to custom YOLO pose model file
        
    Returns:
        Dictionary with keypoint names and positions
    """
    if use_yolo:
        try:
            return _detect_keypoints_yolo(img, confidence, model_path)
        except Exception as e:
            print(f"⚠️  YOLO pose detection failed ({e}), using fallback")
            return _detect_keypoints_fallback(img)
    else:
        return _detect_keypoints_fallback(img)


def _detect_keypoints_yolo(img: np.ndarray, confidence: float = 0.25, model_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Detect body keypoints using YOLOv8 pose estimation.
    
    YOLO pose keypoint indices (COCO format):
    - 5: left_shoulder
    - 6: right_shoulder
    - 11: left_hip
    - 12: right_hip
    
    Args:
        img: Input image (BGR format)
        confidence: Confidence threshold (0.0-1.0)
        model_path: Optional path to custom YOLO pose model
        
    Returns:
        Dictionary with keypoint names and positions
    """
    model = _get_yolo_pose_model(model_path)
    
    # Run YOLO pose detection
    results = model(img, conf=confidence, verbose=False)
    
    keypoints = {}
    height, width = img.shape[:2]
    
    # YOLO pose keypoint indices (COCO format)
    KEYPOINT_INDICES = {
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_hip': 11,
        'right_hip': 12
    }
    
    if results and len(results) > 0:
        result = results[0]
        
        # Check if keypoints are detected
        if result.keypoints is not None and result.keypoints.data is not None:
            # Get the first (most confident) person detected
            keypoints_data = result.keypoints.data
            
            if len(keypoints_data) > 0:
                # Get keypoints for first person (shape: [17, 3] = [x, y, confidence])
                person_keypoints = keypoints_data[0].cpu().numpy()
                
                # Extract needed keypoints
                for name, idx in KEYPOINT_INDICES.items():
                    if idx < len(person_keypoints):
                        kp = person_keypoints[idx]
                        x, y, conf = kp[0], kp[1], kp[2]
                        
                        # Only use keypoint if confidence is reasonable
                        if conf > 0.3:  # Confidence threshold for individual keypoints
                            keypoints[name] = np.array([float(x), float(y)])
    
    # If we didn't get all keypoints, fill in missing ones with fallback
    required_keys = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    if not all(key in keypoints for key in required_keys):
        # Use fallback for missing keypoints
        fallback = _detect_keypoints_fallback(img)
        for key in required_keys:
            if key not in keypoints:
                keypoints[key] = fallback[key]
    
    return keypoints


def _detect_keypoints_fallback(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Fallback: Estimate keypoints based on image dimensions.
    This is a simple placeholder when YOLO pose detection is not available or fails.
    
    Args:
        img: Input image (BGR format)
        
    Returns:
        Dictionary with estimated keypoint positions
    """
    keypoints = {}
    height, width = img.shape[:2]
    
    # Estimate keypoints (center-based estimation)
    # These are rough estimates - not accurate!
    keypoints['left_shoulder'] = np.array([width * 0.4, height * 0.3])
    keypoints['right_shoulder'] = np.array([width * 0.6, height * 0.3])
    keypoints['left_hip'] = np.array([width * 0.4, height * 0.6])
    keypoints['right_hip'] = np.array([width * 0.6, height * 0.6])
    
    return keypoints


def calculate_scaling_factor(
    ref_keypoints: Dict[str, np.ndarray],
    live_keypoints: Dict[str, np.ndarray]
) -> float:
    """
    Calculate scaling factor based on body dimensions.
    
    Args:
        ref_keypoints: Keypoints from reference video
        live_keypoints: Keypoints from live video
        
    Returns:
        Scaling factor (live / reference)
    """
    # Calculate shoulder width for both
    if 'left_shoulder' in ref_keypoints and 'right_shoulder' in ref_keypoints:
        ref_shoulder_width = np.linalg.norm(
            ref_keypoints['right_shoulder'] - ref_keypoints['left_shoulder']
        )
    else:
        ref_shoulder_width = 100.0  # Default
    
    if 'left_shoulder' in live_keypoints and 'right_shoulder' in live_keypoints:
        live_shoulder_width = np.linalg.norm(
            live_keypoints['right_shoulder'] - live_keypoints['left_shoulder']
        )
    else:
        live_shoulder_width = 100.0  # Default
    
    # Calculate torso height for both
    if ('left_shoulder' in ref_keypoints and 'left_hip' in ref_keypoints):
        ref_torso_height = np.linalg.norm(
            ref_keypoints['left_hip'] - ref_keypoints['left_shoulder']
        )
    else:
        ref_torso_height = 100.0  # Default
    
    if ('left_shoulder' in live_keypoints and 'left_hip' in live_keypoints):
        live_torso_height = np.linalg.norm(
            live_keypoints['left_hip'] - live_keypoints['left_shoulder']
        )
    else:
        live_torso_height = 100.0  # Default
    
    # Use average of width and height scaling
    width_scale = live_shoulder_width / ref_shoulder_width if ref_shoulder_width > 0 else 1.0
    height_scale = live_torso_height / ref_torso_height if ref_torso_height > 0 else 1.0
    
    # Return average scaling factor
    return (width_scale + height_scale) / 2.0


def scale_paths(paths: List[np.ndarray], scale_factor: float) -> List[np.ndarray]:
    """
    Scale a list of paths by a scaling factor.
    
    Args:
        paths: List of path arrays
        scale_factor: Scaling factor to apply
        
    Returns:
        List of scaled path arrays
    """
    scaled = []
    for path in paths:
        if len(path) > 0:
            scaled.append(path * scale_factor)
        else:
            scaled.append(path)
    
    return scaled


class BodyScaler:
    """Scales reference paths to match live performer's body size."""
    
    def __init__(self):
        """Initialize body scaler."""
        self.ref_keypoints: Optional[Dict[str, np.ndarray]] = None
    
    def set_reference_keypoints(self, keypoints: Dict[str, np.ndarray]):
        """
        Set reference keypoints.
        
        Args:
            keypoints: Keypoints from reference video
        """
        self.ref_keypoints = keypoints
    
    def scale_reference_paths(
        self,
        reference_paths: List[np.ndarray],
        live_img: np.ndarray
    ) -> List[np.ndarray]:
        """
        Scale reference paths based on live performer's body size.
        
        Args:
            reference_paths: Reference ball paths
            live_img: Current live frame
            
        Returns:
            Scaled reference paths
        """
        if self.ref_keypoints is None:
            # No reference set, return original
            return reference_paths
        
        # Detect live keypoints
        live_keypoints = detect_body_keypoints(live_img)
        
        # Calculate scaling factor
        scale_factor = calculate_scaling_factor(self.ref_keypoints, live_keypoints)
        
        # Scale paths
        return scale_paths(reference_paths, scale_factor)
