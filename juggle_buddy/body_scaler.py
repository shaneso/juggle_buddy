"""
Body scaling module.
Detects body keypoints and scales reference paths to match live performer.
"""
import numpy as np
import cv2
from typing import Dict, List, Optional


def detect_body_keypoints(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Detect body keypoints (shoulders, hips) in an image.
    
    Args:
        img: Input image (BGR format)
        
    Returns:
        Dictionary with keypoint names and positions
    """
    # Use OpenCV DNN for pose estimation (more stable than MediaPipe)
    # For now, return a simple structure
    # In production, would use OpenPose or similar
    
    keypoints = {}
    
    # Placeholder implementation
    # In production, this would use a pose estimation model
    height, width = img.shape[:2]
    
    # Estimate keypoints (simplified - would use actual pose detection)
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
