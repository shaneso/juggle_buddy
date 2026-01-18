"""Script to extract reference ball paths from cascade.mp4"""
import sys
from pathlib import Path

# Add parent directory to Python path so we can import juggle_buddy
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from juggle_buddy.reference_extractor import ReferenceExtractor
from juggle_buddy.reference_extractor import normalize_reference
import numpy as np
import pickle

def main():
    print("=" * 60)
    print("Extracting Reference Ball Paths from cascade.mp4")
    print("=" * 60)
    
    # Create extractor
    extractor = ReferenceExtractor(num_balls=3)
    
    # Extract paths (will use videos/cascade.mp4 by default)
    paths = extractor.extract_from_video()
    
    if paths is None or len(paths) == 0:
        print("ERROR: Failed to extract paths from video")
        return
    
    print(f"\nExtracted {len(paths)} ball paths")
    for i, path in enumerate(paths):
        print(f"  Ball {i}: {len(path)} positions")
    
    # Normalize paths
    normalized = normalize_reference(paths)
    
    # Save reference paths
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "reference_paths.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(normalized, f)
    
    print(f"\n✅ Reference paths saved to: {output_path}")
    print(f"   Total positions: {sum(len(p) for p in normalized)}")
    
    # Display statistics
    print("\nPath Statistics:")
    for i, path in enumerate(normalized):
        if len(path) > 0:
            x_range = path[:, 0].max() - path[:, 0].min()
            y_range = path[:, 1].max() - path[:, 1].min()
            print(f"  Ball {i}: X range: {x_range:.1f}, Y range: {y_range:.1f}")

if __name__ == "__main__":
    main()
