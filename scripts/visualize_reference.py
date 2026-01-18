"""Visualize extracted reference paths"""
import sys
from pathlib import Path

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load reference paths
    data_path = Path("data/reference_paths.pkl")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run extract_reference.py first.")
        return
    
    with open(data_path, 'rb') as f:
        paths = pickle.load(f)
    
    if not paths:
        print("ERROR: No paths loaded. The file may be empty.")
        return
    
    print(f"Loaded {len(paths)} reference paths")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green']
    for i, path in enumerate(paths):
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], 
                   color=colors[i], label=f'Ball {i}', 
                   linewidth=2, alpha=0.7)
        else:
            print(f"Warning: Ball {i} has no positions")
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('Reference Ball Paths (Normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Save plot
    output_path = Path("data/reference_paths_visualization.png")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
