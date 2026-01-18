import cv2
import numpy as np
from scipy.signal import savgol_filter
import tracker

def record_reference(video_path=None):
    """
    Records reference paths from a video or webcam.
    Processes frames, extracts ball positions, smooths them, and saves to .npy
    """
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # Webcam

    red_path = []
    green_path = []
    blue_path = []

    calibration_data = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get calibration data (body scale)
        if calibration_data is None:
            calibration_data = tracker.get_body_scale(frame)
            if calibration_data[0] is None:
                continue  # Wait for pose detection

        # Get ball positions
        positions = tracker.get_ball_positions(frame, calibration_data)
        if positions:
            if 'red' in positions:
                red_path.append(positions['red'])
            if 'green' in positions:
                green_path.append(positions['green'])
            if 'blue' in positions:
                blue_path.append(positions['blue'])

        # Display for monitoring
        cv2.imshow('Recording Reference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Smooth the paths
    def smooth_path(path):
        if len(path) < 10:
            return path
        path = np.array(path)
        smoothed = savgol_filter(path, window_length=11, polyorder=3, axis=0)
        return smoothed.tolist()

    red_path = smooth_path(red_path)
    green_path = smooth_path(green_path)
    blue_path = smooth_path(blue_path)

    # Save to .npy
    reference_data = {
        'red': red_path,
        'green': green_path,
        'blue': blue_path
    }

    np.save('cascade_pattern.npy', reference_data)
    print("Reference data saved to cascade_pattern.npy")

if __name__ == "__main__":
    # For testing, use webcam. Replace with video path if needed.
    record_reference()