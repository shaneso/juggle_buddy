"""
Integration tests for the complete system.
These tests verify that modules work together correctly.
"""
import pytest
import numpy as np
from juggle_buddy.ball_tracker import BallTracker
from juggle_buddy.reference_extractor import ReferenceExtractor
from juggle_buddy.body_scaler import BodyScaler
from juggle_buddy.path_comparator import PathComparator


class TestSystemIntegration:
    """Test integration between modules."""
    
    def test_tracker_to_extractor_flow(self):
        """Test that BallTracker output can be used by ReferenceExtractor."""
        # Create tracker and process some frames
        tracker = BallTracker(num_balls=3)
        
        # Mock some frames (simplified)
        for i in range(10):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            tracker.process_frame(img, frame_number=i)
        
        # Extract paths
        paths = tracker.get_paths()
        
        # Reference extractor should be able to process these
        extractor = ReferenceExtractor()
        reference_paths = extractor.extract_from_tracker(tracker)
        
        assert isinstance(reference_paths, list)
    
    def test_scaler_to_comparator_flow(self):
        """Test that scaled paths can be compared."""
        # Create reference paths
        reference_paths = [
            np.array([[0, 0], [10, 10], [20, 20]]),
            np.array([[30, 30], [40, 40], [50, 50]]),
            np.array([[60, 60], [70, 70], [80, 80]])
        ]
        
        # Scale them
        scale_factor = 1.5
        from juggle_buddy.body_scaler import scale_paths
        scaled_paths = scale_paths(reference_paths, scale_factor)
        
        # Compare scaled paths to original (should have some deviation)
        comparator = PathComparator()
        comparator.set_reference(reference_paths)
        
        scores = comparator.compare(scaled_paths)
        
        assert isinstance(scores, dict)
    
    def test_end_to_end_workflow(self):
        """Test a simplified end-to-end workflow."""
        # 1. Create reference paths (simulated)
        reference_paths = [
            np.array([[100, 200], [110, 210], [120, 220]]),
            np.array([[150, 250], [160, 260], [170, 270]]),
            np.array([[200, 300], [210, 310], [220, 320]])
        ]
        
        # 2. Normalize reference
        from juggle_buddy.reference_extractor import normalize_reference
        normalized_ref = normalize_reference(reference_paths)
        
        # 3. Create live paths (simulated, slightly different)
        live_paths = [
            np.array([[105, 205], [115, 215], [125, 225]]),
            np.array([[155, 255], [165, 265], [175, 275]]),
            np.array([[205, 305], [215, 315], [225, 325]])
        ]
        
        # 4. Scale live paths (simulate body scaling)
        from juggle_buddy.body_scaler import scale_paths
        scaled_live = scale_paths(live_paths, 1.0)  # No scaling for simplicity
        
        # 5. Compare
        comparator = PathComparator()
        comparator.set_reference(normalized_ref)
        scores = comparator.compare(scaled_live)
        
        # Should get valid scores
        assert isinstance(scores, dict)
        # Score should be reasonable (not perfect, but not terrible)
        if 'overall_score' in scores:
            assert 1 <= scores['overall_score'] <= 100
