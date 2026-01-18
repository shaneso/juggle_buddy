"""Main application for Juggle Buddy"""
import cv2
import pickle
import numpy as np
from pathlib import Path
from juggle_buddy.ball_tracker import BallTracker
from juggle_buddy.reference_extractor import ReferenceExtractor, normalize_reference
from juggle_buddy.body_scaler import BodyScaler, detect_body_keypoints, calculate_scaling_factor, scale_paths
from juggle_buddy.path_comparator import PathComparator
from juggle_buddy.video_recorder import VideoRecorder

def load_reference_data():
    """Load pre-extracted reference paths and keypoints."""
    data_dir = Path("data")
    
    # Load reference paths
    ref_paths_file = data_dir / "reference_paths.pkl"
    if not ref_paths_file.exists():
        print("Reference paths not found. Extracting from video...")
        extractor = ReferenceExtractor(num_balls=3)
        paths = extractor.extract_from_video()
        if paths is None:
            return None, None
        normalized = normalize_reference(paths)
        
        # Save for future use
        data_dir.mkdir(exist_ok=True)
        with open(ref_paths_file, 'wb') as f:
            pickle.dump(normalized, f)
    else:
        with open(ref_paths_file, 'rb') as f:
            normalized = pickle.load(f)
    
    # Load reference keypoints
    ref_keypoints_file = data_dir / "reference_keypoints.pkl"
    if ref_keypoints_file.exists():
        with open(ref_keypoints_file, 'rb') as f:
            keypoints = pickle.load(f)
    else:
        keypoints = None
    
    return normalized, keypoints

def main():
    print("=" * 60)
    print("Juggle Buddy - Live Tracking")
    print("=" * 60)
    
    # Load reference data
    print("\nLoading reference data...")
    reference_paths, ref_keypoints = load_reference_data()
    if reference_paths is None:
        print("ERROR: Could not load reference data")
        return
    
    print(f"✅ Loaded {len(reference_paths)} reference paths")
    
    # Initialize components
    tracker = BallTracker(num_balls=3)
    comparator = PathComparator()
    comparator.set_reference(reference_paths)
    
    scaler = BodyScaler()
    if ref_keypoints is not None:
        scaler.set_reference_keypoints(ref_keypoints)
    
    # Open webcam
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Webcam resolution: {frame_width}x{frame_height}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'r' - Reset tracking")
    print("  's' - Show current score")
    
    frame_count = 0
    cycle_length = 90  # Score every 90 frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = tracker.process_frame(frame, frame_count)
        
        # Draw ball detections
        for ball in result['balls']:
            x, y = ball['x'], ball['y']
            r = ball.get('radius', 20)
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        
        # Calculate score every cycle
        if frame_count > 0 and frame_count % cycle_length == 0:
            paths = tracker.get_paths()
            if len(paths) == 3 and all(len(p) > 0 for p in paths):
                # Convert to numpy arrays
                live_paths = [p.positions for p in paths]
                
                # Scale if body scaler is set up
                if ref_keypoints is not None:
                    live_keypoints = detect_body_keypoints(frame)
                    scale_factor = calculate_scaling_factor(ref_keypoints, live_keypoints)
                    live_paths = scale_paths(live_paths, scale_factor)
                
                # Compare
                scores = comparator.compare(live_paths)
                print(f"\nFrame {frame_count} - Score: {scores['overall_score']}/100")
                print(f"  Ball scores: {scores['ball_scores']}")
        
        # Display info
        paths = tracker.get_paths()
        ball_count = sum(1 for p in paths if len(p) > 0)
        cv2.putText(frame, f"Frame: {frame_count} | Balls: {ball_count}/3", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if frame_count > 0 and frame_count % cycle_length == 0:
            paths = tracker.get_paths()
            if len(paths) == 3:
                live_paths = [p.positions for p in paths]
                scores = comparator.compare(live_paths)
                score_text = f"Score: {scores['overall_score']}/100"
                cv2.putText(frame, score_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Juggle Buddy', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker = BallTracker(num_balls=3)
            frame_count = 0
            print("Tracking reset")
        elif key == ord('s'):
            paths = tracker.get_paths()
            if len(paths) == 3:
                live_paths = [p.positions for p in paths]
                scores = comparator.compare(live_paths)
                print(f"\nCurrent Score: {scores['overall_score']}/100")
                print(f"  Ball scores: {scores['ball_scores']}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Tracking complete")

if __name__ == "__main__":
    main()
