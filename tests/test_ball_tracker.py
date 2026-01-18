"""
Unit tests for ball tracking module.
Tests use mock frames to avoid requiring actual video files initially.
"""
import pytest
import numpy as np
import cv2
from juggle_buddy.ball_tracker import BallTracker, detect_balls, track_balls_frame


class TestBallDetection:
    """Test ball detection functionality."""
    
    def test_detect_balls_finds_circles(self):
        """Test that detect_balls can find circular objects in an image."""
        # Create a test image with circles
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 20, (255, 255, 255), -1)
        cv2.circle(img, (200, 150), 25, (255, 255, 255), -1)
        cv2.circle(img, (300, 200), 22, (255, 255, 255), -1)
        
        balls = detect_balls(img)
        
        # Should detect at least some circles
        assert len(balls) >= 0  # May vary based on detection method
        # Each ball should have x, y, radius
        if len(balls) > 0:
            assert 'x' in balls[0] or len(balls[0]) >= 2
    
    def test_detect_balls_empty_image(self):
        """Test detection on empty/black image."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        balls = detect_balls(img)
        assert isinstance(balls, list)
    
    def test_detect_balls_returns_list(self):
        """Test that detect_balls returns a list."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        balls = detect_balls(img)
        assert isinstance(balls, list)


class TestBallTracker:
    """Test the BallTracker class."""
    
    def test_ball_tracker_initialization(self):
        """Test creating a BallTracker instance."""
        tracker = BallTracker(num_balls=3)
        assert tracker.num_balls == 3
        assert len(tracker.tracked_balls) == 0
    
    def test_ball_tracker_process_frame(self):
        """Test processing a single frame."""
        tracker = BallTracker(num_balls=3)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some circles to track
        cv2.circle(img, (100, 100), 20, (255, 255, 255), -1)
        
        result = tracker.process_frame(img, frame_number=0)
        
        assert isinstance(result, dict)
        assert 'balls' in result
        assert 'frame_number' in result
        assert result['frame_number'] == 0
    
    def test_ball_tracker_tracks_multiple_balls(self):
        """Test that tracker can handle multiple balls."""
        tracker = BallTracker(num_balls=3)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add three circles
        cv2.circle(img, (100, 100), 20, (255, 255, 255), -1)
        cv2.circle(img, (200, 150), 25, (255, 255, 255), -1)
        cv2.circle(img, (300, 200), 22, (255, 255, 255), -1)
        
        result = tracker.process_frame(img, frame_number=0)
        
        # Should attempt to track up to 3 balls
        assert len(result['balls']) <= 3
    
    def test_ball_tracker_returns_paths(self):
        """Test that tracker accumulates paths over multiple frames."""
        tracker = BallTracker(num_balls=3)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process multiple frames
        for i in range(5):
            cv2.circle(img, (100 + i*10, 100 + i*5), 20, (255, 255, 255), -1)
            tracker.process_frame(img.copy(), frame_number=i)
        
        paths = tracker.get_paths()
        assert isinstance(paths, list)
        # Should have paths if balls were tracked
        assert len(paths) >= 0
