"""
Unit tests for reference path extraction module.
"""
import pytest
import numpy as np
from juggle_buddy.reference_extractor import ReferenceExtractor, extract_reference_paths, normalize_reference


class TestReferenceExtractor:
    """Test the ReferenceExtractor class."""
    
    def test_reference_extractor_initialization(self):
        """Test creating a ReferenceExtractor instance."""
        extractor = ReferenceExtractor()
        assert extractor.reference_paths is None
        assert extractor.num_balls == 3
    
    def test_extract_reference_paths_from_video(self):
        """Test extracting paths from a video file."""
        extractor = ReferenceExtractor()
        
        # This will need a mock video or test video file
        # For now, test that the method exists and handles missing file gracefully
        try:
            paths = extractor.extract_from_video("nonexistent_video.mp4")
            # If file doesn't exist, should return None or empty
            assert paths is None or len(paths) == 0
        except FileNotFoundError:
            pass  # Expected behavior
    
    def test_extract_reference_paths_from_tracker(self):
        """Test extracting paths from a BallTracker instance."""
        extractor = ReferenceExtractor()
        
        # Mock tracker with paths
        class MockTracker:
            def get_paths(self):
                return [
                    type('BallPath', (), {
                        'ball_id': 0,
                        'positions': np.array([[100, 200], [110, 210], [120, 220]])
                    })(),
                    type('BallPath', (), {
                        'ball_id': 1,
                        'positions': np.array([[150, 250], [160, 260], [170, 270]])
                    })(),
                    type('BallPath', (), {
                        'ball_id': 2,
                        'positions': np.array([[200, 300], [210, 310], [220, 320]])
                    })()
                ]
        
        tracker = MockTracker()
        paths = extractor.extract_from_tracker(tracker)
        
        assert len(paths) == 3
        assert all(isinstance(p, np.ndarray) for p in paths)
        assert all(p.shape[1] == 2 for p in paths)  # Each path has (x, y) coordinates


class TestNormalizeReference:
    """Test reference normalization functions."""
    
    def test_normalize_reference_centers_paths(self):
        """Test that normalize_reference centers all paths."""
        paths = [
            np.array([[100, 200], [110, 210]]),
            np.array([[150, 250], [160, 260]]),
            np.array([[200, 300], [210, 310]])
        ]
        
        normalized = normalize_reference(paths)
        
        # All paths should be centered around origin
        for path in normalized:
            center = path.mean(axis=0)
            assert np.allclose(center, [0, 0], atol=10.0)  # Allow some tolerance
    
    def test_normalize_reference_preserves_relative_positions(self):
        """Test that normalization preserves relative ball positions."""
        paths = [
            np.array([[100, 200], [110, 210]]),
            np.array([[150, 250], [160, 260]])
        ]
        
        # Calculate relative distance before normalization
        original_distance = np.linalg.norm(paths[0][0] - paths[1][0])
        
        normalized = normalize_reference(paths)
        
        # Relative distance should be preserved (approximately)
        normalized_distance = np.linalg.norm(normalized[0][0] - normalized[1][0])
        assert normalized_distance == pytest.approx(original_distance, rel=0.1)
