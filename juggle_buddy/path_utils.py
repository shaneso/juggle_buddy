"""
Path utility functions for ball path data structures and operations.
"""
import numpy as np
from typing import List, Tuple


class BallPath:
    """Represents a path taken by a single ball."""
    
    def __init__(self, ball_id: int, positions: np.ndarray):
        """
        Initialize a BallPath.
        
        Args:
            ball_id: Unique identifier for the ball
            positions: Array of shape (N, 2) containing (x, y) positions
        """
        self.ball_id = ball_id
        self.positions = positions
    
    def __len__(self) -> int:
        """Return the number of positions in the path."""
        return len(self.positions)
    
    def get_position(self, frame_index: int) -> np.ndarray:
        """
        Get position at a specific frame index.
        
        Args:
            frame_index: Index of the frame
            
        Returns:
            Array of shape (2,) containing (x, y) position
        """
        if frame_index < 0 or frame_index >= len(self.positions):
            raise IndexError(f"Frame index {frame_index} out of range")
        return self.positions[frame_index]


def normalize_path(positions: np.ndarray) -> np.ndarray:
    """
    Normalize a path by centering it around the origin.
    
    Args:
        positions: Array of shape (N, 2) containing (x, y) positions
        
    Returns:
        Normalized positions array of same shape
    """
    if len(positions) == 0:
        return positions
    
    # Calculate center
    center = positions.mean(axis=0)
    
    # Center around origin
    normalized = positions - center
    
    return normalized


def calculate_path_length(positions: np.ndarray) -> float:
    """
    Calculate the total length of a path.
    
    Args:
        positions: Array of shape (N, 2) containing (x, y) positions
        
    Returns:
        Total path length
    """
    if len(positions) <= 1:
        return 0.0
    
    # Calculate distances between consecutive points
    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    
    return float(np.sum(distances))
