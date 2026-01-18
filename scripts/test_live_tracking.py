"""Test live ball tracking from webcam"""
import sys
from pathlib import Path

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from juggle_buddy.ball_tracker import BallTracker, detect_balls

def main():
    print("Starting webcam...")
    print("Press 'd' to toggle debug mask view, 'q' to quit, 's' to save current paths")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    tracker = BallTracker(num_balls=3)
    frame_count = 0
    show_debug_mask = False
    
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
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Debug mode: show color mask
        if show_debug_mask:
            # Load config and create mask
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            config_path = Path(__file__).parent.parent / "data" / "ball_color_config.py"
            
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            if config_path.exists():
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("ball_color_config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    
                    if hasattr(config_module, 'BALL_COLOR_RANGES'):
                        color_ranges = config_module.BALL_COLOR_RANGES
                        for color_name, ranges in color_ranges.items():
                            if isinstance(ranges, tuple):
                                lower, upper = ranges
                                mask = cv2.inRange(hsv, lower, upper)
                                combined_mask = cv2.bitwise_or(combined_mask, mask)
                except Exception as e:
                    print(f"Debug: Could not load config: {e}")
            
            # Apply same morphological operations as in detection
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
            
            mask_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            display = np.hstack([frame, mask_bgr])
            cv2.putText(display, "LEFT: Original | RIGHT: Color Mask (White = Detected)", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Live Tracking', display)
        else:
            cv2.imshow('Live Tracking', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_debug_mask = not show_debug_mask
            print(f"Debug mask: {'ON' if show_debug_mask else 'OFF'}")
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
