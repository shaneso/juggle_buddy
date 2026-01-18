"""Extract body keypoints from reference video"""
import sys
from pathlib import Path

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from juggle_buddy.body_scaler import detect_body_keypoints
import pickle

def main():
    video_path = Path("videos/cascade.mp4")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return
    
    # Get a representative frame (middle of video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame")
        return
    
    print(f"Extracting keypoints from frame {mid_frame} of {total_frames}")
    
    # Detect keypoints
    keypoints = detect_body_keypoints(frame)
    
    print("\nDetected keypoints:")
    for name, pos in keypoints.items():
        print(f"  {name}: ({pos[0]:.1f}, {pos[1]:.1f})")
    
    # Save keypoints
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "reference_keypoints.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(keypoints, f)
    
    print(f"\n✅ Reference keypoints saved to: {output_path}")
    
    # Visualize
    vis_frame = frame.copy()
    for name, pos in keypoints.items():
        x, y = int(pos[0]), int(pos[1])
        cv2.circle(vis_frame, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(vis_frame, name, (x+10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    vis_path = output_dir / "reference_keypoints_visualization.jpg"
    cv2.imwrite(str(vis_path), vis_frame)
    print(f"✅ Visualization saved to: {vis_path}")
    
    cap.release()

if __name__ == "__main__":
    main()
