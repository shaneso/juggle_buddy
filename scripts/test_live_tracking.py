"""Test live ball tracking from webcam"""
import sys
from pathlib import Path

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from juggle_buddy.ball_tracker import BallTracker

def main():
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    tracker = BallTracker(num_balls=3)
    frame_count = 0
    
    print("Press 'q' to quit, 's' to save current paths")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = tracker.process_frame(frame, frame_count)
        
        # Draw detections
        for ball in result['balls']:
            x, y = ball['x'], ball['y']
            r = ball.get('radius', 20)
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        
        # Display frame number and ball count
        cv2.putText(frame, f"Frame: {frame_count}, Balls: {len(result['balls'])}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Live Tracking', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            paths = tracker.get_paths()
            print(f"\nSaved {len(paths)} paths")
            for i, path in enumerate(paths):
                print(f"  Ball {i}: {len(path)} positions")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final paths
    paths = tracker.get_paths()
    print(f"\nFinal tracking results:")
    print(f"  Total frames: {frame_count}")
    print(f"  Tracked balls: {len(paths)}")
    for i, path in enumerate(paths):
        print(f"  Ball {i}: {len(path)} positions")

if __name__ == "__main__":
    main()
