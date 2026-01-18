"""Test path comparison with reference paths"""
import sys
from pathlib import Path

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pickle
import numpy as np
from juggle_buddy.path_comparator import PathComparator

def main():
    # Load reference paths
    ref_path = Path("data/reference_paths.pkl")
    if not ref_path.exists():
        print(f"ERROR: {ref_path} not found. Run extract_reference.py first.")
        return
    
    with open(ref_path, 'rb') as f:
        reference_paths = pickle.load(f)
    
    print(f"Loaded {len(reference_paths)} reference paths")
    
    # Create comparator
    comparator = PathComparator()
    comparator.set_reference(reference_paths)
    
    # Test 1: Perfect match
    print("\nTest 1: Perfect match")
    perfect_paths = [path.copy() for path in reference_paths]
    scores = comparator.compare(perfect_paths)
    print(f"  Overall score: {scores['overall_score']}")
    print(f"  Ball scores: {scores['ball_scores']}")
    
    # Test 2: Slight deviation
    print("\nTest 2: Slight deviation (+10 pixels)")
    deviated_paths = [path + 10 for path in reference_paths]
    scores = comparator.compare(deviated_paths)
    print(f"  Overall score: {scores['overall_score']}")
    print(f"  Ball scores: {scores['ball_scores']}")
    
    # Test 3: Large deviation
    print("\nTest 3: Large deviation (+50 pixels)")
    large_dev_paths = [path + 50 for path in reference_paths]
    scores = comparator.compare(large_dev_paths)
    print(f"  Overall score: {scores['overall_score']}")
    print(f"  Ball scores: {scores['ball_scores']}")

if __name__ == "__main__":
    main()
