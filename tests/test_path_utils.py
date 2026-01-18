"""
Unit tests for path utility functions.
These are the simplest components - data structures and basic operations.
"""
import pytest
import numpy as np
from juggle_buddy.path_utils import BallPath, normalize_path, calculate_path_length


class TestBallPath:
    """Test the BallPath data structure."""
    
    def test_ball_path_creation(self):
        """Test creating a BallPath object."""
        path = BallPath(ball_id=1, positions=np.array([[10, 20], [15, 25], [20, 30]]))
        assert path.ball_id == 1
        assert path.positions.shape == (3, 2)
        assert len(path) == 3
    
    def test_ball_path_empty(self):
        """Test creating an empty BallPath."""
        path = BallPath(ball_id=0, positions=np.array([]).reshape(0, 2))
        assert len(path) == 0
    
    def test_ball_path_get_position(self):
        """Test getting position at specific frame."""
        path = BallPath(ball_id=1, positions=np.array([[10, 20], [15, 25], [20, 30]]))
        assert np.array_equal(path.get_position(0), np.array([10, 20]))
        assert np.array_equal(path.get_position(2), np.array([20, 30]))


class TestNormalizePath:
    """Test path normalization functions."""
    
    def test_normalize_path_centers_around_origin(self):
        """Test that normalize_path centers the path around (0, 0)."""
        positions = np.array([[100, 200], [110, 210], [120, 220]])
        normalized = normalize_path(positions)
        
        # Center should be approximately at origin
        center = normalized.mean(axis=0)
        assert np.allclose(center, [0, 0], atol=1.0)
    
    def test_normalize_path_preserves_shape(self):
        """Test that normalization preserves path shape."""
        positions = np.array([[100, 200], [110, 210], [120, 220]])
        normalized = normalize_path(positions)
        
        assert normalized.shape == positions.shape
    
    def test_normalize_path_empty(self):
        """Test normalizing an empty path."""
        positions = np.array([]).reshape(0, 2)
        normalized = normalize_path(positions)
        assert normalized.shape == (0, 2)


class TestCalculatePathLength:
    """Test path length calculations."""
    
    def test_calculate_path_length_simple(self):
        """Test calculating length of a simple path."""
        positions = np.array([[0, 0], [10, 0], [20, 0]])  # Horizontal line
        length = calculate_path_length(positions)
        assert length == pytest.approx(20.0, abs=0.1)
    
    def test_calculate_path_length_diagonal(self):
        """Test calculating length of a diagonal path."""
        positions = np.array([[0, 0], [10, 10]])  # Diagonal
        length = calculate_path_length(positions)
        assert length == pytest.approx(14.14, abs=0.1)  # sqrt(200)
    
    def test_calculate_path_length_empty(self):
        """Test calculating length of empty path."""
        positions = np.array([]).reshape(0, 2)
        length = calculate_path_length(positions)
        assert length == 0.0
    
    def test_calculate_path_length_single_point(self):
        """Test calculating length of single point path."""
        positions = np.array([[10, 20]])
        length = calculate_path_length(positions)
        assert length == 0.0
