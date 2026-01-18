"""Calibration tool to find HSV color ranges for red, blue, and green balls"""
import sys
from pathlib import Path

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from pathlib import Path

# Default HSV ranges for red, blue, green (will be adjusted)
COLOR_RANGES = {
    'red': [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),  # Red range 1 (hue 0-10)
        (np.array([170, 120, 70]), np.array([180, 255, 255]))  # Red range 2 (hue 170-180)
    ],
    'blue': [(np.array([100, 150, 0]), np.array([124, 255, 255]))],
    'green': [(np.array([35, 50, 50]), np.array([85, 255, 255]))]
}

def nothing(x):
    """Callback for trackbar (does nothing)."""
    pass

def calibrate_color(color_name: str, ranges: list):
    """Interactive calibration for a single color."""
    print(f"\n{'='*60}")
    print(f"Calibrating {color_name.upper()} ball detection")
    print(f"{'='*60}")
    print("Instructions:")
    print("1. Hold a {color_name} ball in front of the camera")
    print("2. Adjust trackbars until the ball is highlighted in white")
    print("3. Press 's' to save these values")
    print("4. Press 'q' to quit")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return None
    
    # Create window and trackbars
    window_name = f'Calibrate {color_name}'
    cv2.namedWindow(window_name)
    
    # Create trackbars for first range
    cv2.createTrackbar('H Min', window_name, ranges[0][0][0], 179, nothing)
    cv2.createTrackbar('S Min', window_name, ranges[0][0][1], 255, nothing)
    cv2.createTrackbar('V Min', window_name, ranges[0][0][2], 255, nothing)
    cv2.createTrackbar('H Max', window_name, ranges[0][1][0], 179, nothing)
    cv2.createTrackbar('S Max', window_name, ranges[0][1][1], 255, nothing)
    cv2.createTrackbar('V Max', window_name, ranges[0][1][2], 255, nothing)
    
    saved_range = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get trackbar values
        h_min = cv2.getTrackbarPos('H Min', window_name)
        s_min = cv2.getTrackbarPos('S Min', window_name)
        v_min = cv2.getTrackbarPos('V Min', window_name)
        h_max = cv2.getTrackbarPos('H Max', window_name)
        s_max = cv2.getTrackbarPos('S Max', window_name)
        v_max = cv2.getTrackbarPos('V Max', window_name)
        
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        
        # Convert to HSV and create mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        
        # For red, also check second range if provided (red wraps around hue 0/180)
        # Note: You can adjust the first range to cover both ranges, or calibrate separately
        if color_name == 'red' and len(ranges) > 1:
            # Optionally combine with second range (uncomment if needed):
            # lower2, upper2 = ranges[1]
            # mask2 = cv2.inRange(hsv, lower2, upper2)
            # mask = cv2.bitwise_or(mask, mask2)
            pass
        
        # Apply mask - white pixels show detected color
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Create a better visualization: show original, mask (white = detected), and result
        # Convert mask to 3-channel for display
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Add text overlay showing current HSV values
        text_frame = frame.copy()
        cv2.putText(text_frame, f"HSV: [{h_min}, {s_min}, {v_min}] to [{h_max}, {s_max}, {v_max}]",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(text_frame, "White pixels = detected color", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        text_mask = mask_bgr.copy()
        cv2.putText(text_mask, "MASK VIEW", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combine: original | mask | result
        # Resize if needed to fit on screen
        h, w = frame.shape[:2]
        max_width = 1400
        if w * 3 > max_width:
            scale = max_width / (w * 3)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame_resized = cv2.resize(text_frame, (new_w, new_h))
            mask_resized = cv2.resize(text_mask, (new_w, new_h))
            result_resized = cv2.resize(result, (new_w, new_h))
            display = np.hstack([frame_resized, mask_resized, result_resized])
        else:
            display = np.hstack([text_frame, text_mask, result])
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            saved_range = (lower.copy(), upper.copy())
            print(f"\n✅ Saved {color_name} range:")
            print(f"   Lower: [{h_min}, {s_min}, {v_min}]")
            print(f"   Upper: [{h_max}, {s_max}, {v_max}]")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return saved_range

def main():
    print("=" * 60)
    print("Ball Color Calibration Tool")
    print("=" * 60)
    print("\nThis tool helps you find the right HSV color ranges")
    print("for detecting your red, blue, and green balls.")
    
    calibrated = {}
    
    for color in ['red', 'blue', 'green']:
        range_result = calibrate_color(color, COLOR_RANGES[color])
        if range_result:
            calibrated[color] = range_result
            print(f"\n{color} calibration complete!")
        else:
            print(f"\n⚠️  {color} calibration skipped or failed")
    
    if calibrated:
        # Save to config file
        config_path = Path("data/ball_color_config.py")
        config_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(config_path, 'w') as f:
            f.write("# Ball color HSV ranges (calibrated)\n")
            f.write("# Automatically loaded by juggle_buddy/ball_tracker.py\n\n")
            f.write("import numpy as np\n\n")
            f.write("BALL_COLOR_RANGES = {\n")
            for color, (lower, upper) in calibrated.items():
                f.write(f"    '{color}': (np.array({lower.tolist()}), np.array({upper.tolist()})),\n")
            f.write("}\n")
        
        print(f"\n✅ Calibration saved to: {config_path}")
        print("   You can now use these values in ball_tracker.py")
    else:
        print("\n⚠️  No calibrations were saved")

if __name__ == "__main__":
    main()
