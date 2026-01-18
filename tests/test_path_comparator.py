"""
Unit tests for path comparison and scoring module.
"""
import pytest
import numpy as np
from juggle_buddy.path_comparator import PathComparator, align_paths, calculate_deviation, compute_score


class TestAlignPaths:
    """Test path alignment functionality."""
    
    def test_align_paths_same_length(self):
        """Test aligning paths of the same length."""
        reference = np.array([[0, 0], [10, 10], [20, 20]])
        live = np.array([[1, 1], [11, 11], [21, 21]])
        
        aligned = align_paths(reference, live)
        
        assert aligned.shape == reference.shape
    
    def test_align_paths_different_length(self):
        """Test aligning paths of different lengths."""
        reference = np.array([[0, 0], [10, 10], [20, 20], [30, 30]])
        live = np.array([[1, 1], [11, 11], [21, 21]])
        
        aligned = align_paths(reference, live)
        
        # Should handle length mismatch (interpolate or truncate)
        assert aligned.shape[0] == reference.shape[0] or aligned.shape[0] == live.shape[0]
    
    def test_align_paths_finds_best_offset(self):
        """Test that alignment finds the best temporal offset."""
        # Create a reference path
        reference = np.array([[0, 0], [10, 0], [20, 0], [30, 0]])
        
        # Create a live path that's shifted by 1 frame
        live = np.array([[10, 0], [20, 0], [30, 0], [40, 0]])
        
        aligned = align_paths(reference, live)
        
        # Should align them properly
        assert aligned.shape == reference.shape


class TestCalculateDeviation:
    """Test deviation calculation."""
    
    def test_calculate_deviation_perfect_match(self):
        """Test deviation when paths match perfectly."""
        reference = np.array([[0, 0], [10, 10], [20, 20]])
        live = np.array([[0, 0], [10, 10], [20, 20]])
        
        deviation = calculate_deviation(reference, live)
        
        assert deviation['horizontal'] == pytest.approx(0.0, abs=0.1)
        assert deviation['vertical'] == pytest.approx(0.0, abs=0.1)
        assert deviation['total'] == pytest.approx(0.0, abs=0.1)
    
    def test_calculate_deviation_horizontal_offset(self):
        """Test deviation calculation with horizontal offset."""
        reference = np.array([[0, 0], [10, 0], [20, 0]])
        live = np.array([[5, 0], [15, 0], [25, 0]])  # Shifted right by 5
        
        deviation = calculate_deviation(reference, live)
        
        assert deviation['horizontal'] > 0
        assert deviation['vertical'] == pytest.approx(0.0, abs=0.1)
    
    def test_calculate_deviation_vertical_offset(self):
        """Test deviation calculation with vertical offset."""
        reference = np.array([[0, 0], [0, 10], [0, 20]])
        live = np.array([[0, 5], [0, 15], [0, 25]])  # Shifted down by 5
        
        deviation = calculate_deviation(reference, live)
        
        assert deviation['vertical'] > 0
        assert deviation['horizontal'] == pytest.approx(0.0, abs=0.1)
    
    def test_calculate_deviation_returns_dict(self):
        """Test that deviation calculation returns expected structure."""
        reference = np.array([[0, 0], [10, 10]])
        live = np.array([[1, 1], [11, 11]])
        
        deviation = calculate_deviation(reference, live)
        
        assert isinstance(deviation, dict)
        assert 'horizontal' in deviation
        assert 'vertical' in deviation
        assert 'total' in deviation


class TestComputeScore:
    """Test score computation."""
    
    def test_compute_score_perfect_match(self):
        """Test score for perfect path match."""
        reference = np.array([[0, 0], [10, 10], [20, 20]])
        live = np.array([[0, 0], [10, 10], [20, 20]])
        
        score = compute_score(reference, live)
        
        assert score == 100
    
    def test_compute_score_large_deviation(self):
        """Test score for large deviation."""
        reference = np.array([[0, 0], [10, 10], [20, 20]])
        live = np.array([[100, 100], [110, 110], [120, 120]])  # Very far away
        
        score = compute_score(reference, live)
        
        assert 1 <= score < 100  # Should be low but not below 1
    
    def test_compute_score_in_range(self):
        """Test that score is always in 1-100 range."""
        reference = np.array([[0, 0], [10, 10], [20, 20]])
        
        # Test various deviations
        for offset in [0, 5, 10, 50, 100]:
            live = reference + offset
            score = compute_score(reference, live)
            assert 1 <= score <= 100
    
    def test_compute_score_handles_empty_paths(self):
        """Test score computation with empty paths."""
        reference = np.array([]).reshape(0, 2)
        live = np.array([]).reshape(0, 2)
        
        score = compute_score(reference, live)
        
        # Should return a default score or handle gracefully
        assert 1 <= score <= 100 or score is None


class TestPathComparator:
    """Test the PathComparator class."""
    
    def test_path_comparator_initialization(self):
        """Test creating a PathComparator instance."""
        comparator = PathComparator()
        assert comparator.reference_paths is None
    
    def test_path_comparator_set_reference(self):
        """Test setting reference paths."""
        comparator = PathComparator()
        reference_paths = [
            np.array([[0, 0], [10, 10]]),
            np.array([[20, 20], [30, 30]]),
            np.array([[40, 40], [50, 50]])
        ]
        
        comparator.set_reference(reference_paths)
        
        assert len(comparator.reference_paths) == 3
    
    def test_path_comparator_compare_paths(self):
        """Test comparing live paths to reference."""
        comparator = PathComparator()
        reference_paths = [
            np.array([[0, 0], [10, 10]]),
            np.array([[20, 20], [30, 30]]),
            np.array([[40, 40], [50, 50]])
        ]
        comparator.set_reference(reference_paths)
        
        live_paths = [
            np.array([[1, 1], [11, 11]]),
            np.array([[21, 21], [31, 31]]),
            np.array([[41, 41], [51, 51]])
        ]
        
        scores = comparator.compare(live_paths)
        
        assert isinstance(scores, dict)
        assert 'ball_scores' in scores or 'overall_score' in scores
        assert len(scores.get('ball_scores', [])) == 3 or 'overall_score' in scores
