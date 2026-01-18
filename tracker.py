import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_body_scale(frame):
    """
    Uses MediaPipe Pose to detect body landmarks and return shoulder width and hip center.
    Returns: shoulder_width (pixels), hip_center (x, y)
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Shoulder landmarks: 11 (left), 12 (right)
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Hip landmarks: 23 (left), 24 (right)
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculate shoulder width
        shoulder_width = abs(right_shoulder.x - left_shoulder.x) * frame.shape[1]

        # Hip center
        hip_center_x = ((left_hip.x + right_hip.x) / 2) * frame.shape[1]
        hip_center_y = ((left_hip.y + right_hip.y) / 2) * frame.shape[0]

        return shoulder_width, (hip_center_x, hip_center_y)
    else:
        return None, None

def get_ball_positions(frame, calibration_data):
    """
    Detects ball positions for red, green, blue balls.
    Normalizes coordinates based on calibration_data (shoulder_width, hip_center).
    Returns: {'red': (x,y), 'green': (x,y), 'blue': (x,y)} normalized
    """
    if calibration_data is None:
        return None

    shoulder_width, hip_center = calibration_data
    if shoulder_width is None or hip_center is None:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges (these may need tuning)
    color_ranges = {
        'red': [(0, 50, 50), (10, 255, 255)],  # Lower red
        'green': [(40, 50, 50), (80, 255, 255)],
        'blue': [(90, 50, 50), (130, 255, 255)]
    }

    positions = {}

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Normalize
                norm_x = (cx - hip_center[0]) / shoulder_width
                norm_y = (cy - hip_center[1]) / shoulder_width

                positions[color] = (norm_x, norm_y)

    return positions