"""
Unit tests for body scaling module.
"""
import pytest
import numpy as np
from juggle_buddy.body_scaler import BodyScaler, detect_body_keypoints, calculate_scaling_factor, scale_paths


class TestBodyKeypointDetection:
    """Test body keypoint detection."""
    
    def test_detect_body_keypoints_returns_keypoints(self):
        """Test that keypoint detection returns expected structure."""
        # Create a simple test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        keypoints = detect_body_keypoints(img)
        
        # Should return a dict with keypoint locations
        assert isinstance(keypoints, dict)
        # Should have at least shoulders and hips
        assert 'left_shoulder' in keypoints or 'shoulders' in keypoints
        assert 'left_hip' in keypoints or 'hips' in keypoints
    
    def test_detect_body_keypoints_handles_empty_image(self):
        """Test keypoint detection on empty image."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = detect_body_keypoints(img)
        assert isinstance(keypoints, dict)


class TestCalculateScalingFactor:
    """Test scaling factor calculation."""
    
    def test_calculate_scaling_factor_same_size(self):
        """Test scaling factor when bodies are same size."""
        ref_keypoints = {
            'left_shoulder': np.array([100, 100]),
            'right_shoulder': np.array([200, 100]),
            'left_hip': np.array([100, 300]),
            'right_hip': np.array([200, 300])
        }
        
        live_keypoints = {
            'left_shoulder': np.array([100, 100]),
            'right_shoulder': np.array([200, 100]),
            'left_hip': np.array([100, 300]),
            'right_hip': np.array([200, 300])
        }
        
        scale = calculate_scaling_factor(ref_keypoints, live_keypoints)
        assert scale == pytest.approx(1.0, abs=0.1)
    
    def test_calculate_scaling_factor_different_size(self):
        """Test scaling factor when bodies are different sizes."""
        ref_keypoints = {
            'left_shoulder': np.array([100, 100]),
            'right_shoulder': np.array([200, 100]),
            'left_hip': np.array([100, 300]),
            'right_hip': np.array([200, 300])
        }
        
        # Live person is twice as wide
        live_keypoints = {
            'left_shoulder': np.array([50, 100]),
            'right_shoulder': np.array([250, 100]),
            'left_hip': np.array([50, 300]),
            'right_hip': np.array([250, 300])
        }
        
        scale = calculate_scaling_factor(ref_keypoints, live_keypoints)
        assert scale == pytest.approx(2.0, abs=0.2)
    
    def test_calculate_scaling_factor_handles_missing_keypoints(self):
        """Test that scaling factor calculation handles missing keypoints."""
        ref_keypoints = {'left_shoulder': np.array([100, 100])}
        live_keypoints = {'left_shoulder': np.array([200, 100])}
        
        # Should return a default scale or handle gracefully
        scale = calculate_scaling_factor(ref_keypoints, live_keypoints)
        assert scale > 0


class TestScalePaths:
    """Test path scaling functionality."""
    
    def test_scale_paths_applies_scaling(self):
        """Test that scale_paths correctly scales path coordinates."""
        paths = [
            np.array([[10, 20], [15, 25]]),
            np.array([[30, 40], [35, 45]])
        ]
        
        scale_factor = 2.0
        scaled = scale_paths(paths, scale_factor)
        
        # Check that coordinates are scaled
        assert np.allclose(scaled[0][0], [20, 40], atol=0.1)
        assert np.allclose(scaled[1][0], [60, 80], atol=0.1)
    
    def test_scale_paths_preserves_shape(self):
        """Test that scaling preserves path shapes."""
        paths = [
            np.array([[10, 20], [15, 25]]),
            np.array([[30, 40]])
        ]
        
        scaled = scale_paths(paths, 1.5)
        
        assert scaled[0].shape == paths[0].shape
        assert scaled[1].shape == paths[1].shape
    
    def test_scale_paths_empty_list(self):
        """Test scaling empty path list."""
        paths = []
        scaled = scale_paths(paths, 2.0)
        assert scaled == []
