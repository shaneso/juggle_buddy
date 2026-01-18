"""Test ball detection on a single frame from reference video"""
import sys
from pathlib import Path

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from juggle_buddy.ball_tracker import detect_balls

def main():
    video_path = Path("videos/cascade.mp4")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return
    
    # Create output directory
    output_dir = Path("data/detection_test")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Read a few frames and test detection
    for frame_num in [0, 30, 60, 90]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        print(f"\nFrame {frame_num}:")
        balls = detect_balls(frame)
        print(f"  Detected {len(balls)} balls")
        
        # Visualize detections
        vis_frame = frame.copy()
        for ball in balls:
            x, y = ball['x'], ball['y']
            r = ball.get('radius', 20)
            cv2.circle(vis_frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis_frame, (x, y), 2, (0, 0, 255), -1)
        
        # Save visualization
        output_path = output_dir / f"frame_{frame_num:05d}.jpg"
        cv2.imwrite(str(output_path), vis_frame)
        print(f"  Saved to: {output_path}")
    
    cap.release()
    print(f"\n✅ Test complete! Check {output_dir} for results")

if __name__ == "__main__":
    main()
