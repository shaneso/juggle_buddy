import cv2
import numpy as np
from scipy.spatial import KDTree
import tracker

def load_reference(pattern_file='cascade_pattern.npy'):
    """
    Loads reference data and builds KD-Trees for each color.
    """
    data = np.load(pattern_file, allow_pickle=True).item()
    trees = {}
    for color, path in data.items():
        if path:
            trees[color] = KDTree(path)
    return trees, data

def denormalize_point(point, calibration_data):
    """
    Converts normalized point back to pixel coordinates.
    """
    shoulder_width, hip_center = calibration_data
    x = point[0] * shoulder_width + hip_center[0]
    y = point[1] * shoulder_width + hip_center[1]
    return int(x), int(y)

def main():
    # Load reference data
    trees, reference_data = load_reference()

    cap = cv2.VideoCapture(0)
    calibration_data = None

    # For scoring
    scores = []
    sensitivity = 500

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get calibration
        if calibration_data is None:
            calibration_data = tracker.get_body_scale(frame)

        # Get current positions
        positions = tracker.get_ball_positions(frame, calibration_data)
        if positions and calibration_data:
            total_deviation = 0
            for color in ['red', 'green', 'blue']:
                if color in positions and color in trees:
                    dist, _ = trees[color].query(positions[color])
                    total_deviation += dist

                    # Visualize
                    cx, cy = denormalize_point(positions[color], calibration_data)
                    if dist < 0.1:
                        cv2.circle(frame, (cx, cy), 20, (0, 255, 0), 2)  # Green for good
                    else:
                        cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 2)  # Red for bad

            # Calculate score
            instant_score = max(0, 100 - (total_deviation * sensitivity))
            scores.append(instant_score)
            if len(scores) > 90:
                scores.pop(0)
            session_score = np.mean(scores) if scores else 0

            # Display score
            cv2.putText(frame, f"Score: {session_score:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw ghost paths
        if calibration_data and reference_data:
            for color, path in reference_data.items():
                color_map = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)}
                for point in path[::5]:  # Every 5th point for performance
                    px, py = denormalize_point(point, calibration_data)
                    cv2.circle(frame, (px, py), 2, color_map.get(color, (128, 128, 128)), -1)

        cv2.imshow('Juggle Buddy', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()