"""
Path comparison and scoring module.
Compares live ball paths to reference paths and computes scores.
"""
import numpy as np
from typing import List, Dict, Optional


def align_paths(reference: np.ndarray, live: np.ndarray) -> np.ndarray:
    """
    Align live path with reference path (temporal alignment).
    
    Args:
        reference: Reference path array
        live: Live path array
        
    Returns:
        Aligned live path
    """
    if len(reference) == 0 or len(live) == 0:
        return live
    
    # If same length, return as-is
    if len(reference) == len(live):
        return live
    
    # If live is shorter, pad with last value
    if len(live) < len(reference):
        padding = np.tile(live[-1:], (len(reference) - len(live), 1))
        return np.vstack([live, padding])
    
    # If live is longer, truncate
    return live[:len(reference)]


def calculate_deviation(
    reference: np.ndarray,
    live: np.ndarray
) -> Dict[str, float]:
    """
    Calculate deviation between reference and live paths.
    
    Args:
        reference: Reference path array
        live: Live path array (should be aligned)
        
    Returns:
        Dictionary with 'horizontal', 'vertical', and 'total' deviations
    """
    if len(reference) == 0 or len(live) == 0:
        return {'horizontal': 0.0, 'vertical': 0.0, 'total': 0.0}
    
    # Align paths
    aligned_live = align_paths(reference, live)
    
    # Calculate differences
    diff = aligned_live - reference
    
    # Calculate deviations
    horizontal_dev = np.mean(np.abs(diff[:, 0]))
    vertical_dev = np.mean(np.abs(diff[:, 1]))
    total_dev = np.mean(np.linalg.norm(diff, axis=1))
    
    return {
        'horizontal': float(horizontal_dev),
        'vertical': float(vertical_dev),
        'total': float(total_dev)
    }


def compute_score(reference: np.ndarray, live: np.ndarray) -> int:
    """
    Compute a score (1-100) based on path deviation.
    
    Args:
        reference: Reference path array
        live: Live path array
        
    Returns:
        Score from 1 to 100
    """
    if len(reference) == 0 or len(live) == 0:
        return 50  # Default score for empty paths
    
    # Calculate deviation
    deviation = calculate_deviation(reference, live)
    total_dev = deviation['total']
    
    # Normalize deviation (this is a heuristic - may need tuning)
    # Assume typical path spans ~200 pixels, so max reasonable deviation is ~100
    max_deviation = 100.0
    normalized_dev = min(total_dev / max_deviation, 1.0)
    
    # Convert to score (100 = perfect, 1 = worst)
    score = int(100 * (1.0 - normalized_dev))
    
    # Clamp to 1-100 range
    return max(1, min(100, score))


class PathComparator:
    """Compares live paths to reference paths and computes scores."""
    
    def __init__(self):
        """Initialize path comparator."""
        self.reference_paths: Optional[List[np.ndarray]] = None
    
    def set_reference(self, reference_paths: List[np.ndarray]):
        """
        Set reference paths.
        
        Args:
            reference_paths: List of reference path arrays
        """
        self.reference_paths = reference_paths
    
    def compare(self, live_paths: List[np.ndarray]) -> Dict:
        """
        Compare live paths to reference paths.
        
        Args:
            live_paths: List of live path arrays
            
        Returns:
            Dictionary with scores for each ball and overall score
        """
        if self.reference_paths is None:
            raise ValueError("Reference paths not set")
        
        if len(live_paths) != len(self.reference_paths):
            raise ValueError(
                f"Number of live paths ({len(live_paths)}) doesn't match "
                f"reference paths ({len(self.reference_paths)})"
            )
        
        ball_scores = []
        for ref_path, live_path in zip(self.reference_paths, live_paths):
            score = compute_score(ref_path, live_path)
            ball_scores.append(score)
        
        # Calculate overall score (average)
        overall_score = int(np.mean(ball_scores)) if ball_scores else 50
        
        return {
            'ball_scores': ball_scores,
            'overall_score': overall_score
        }
