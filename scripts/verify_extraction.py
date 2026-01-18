"""Verify extracted reference paths by showing detected balls on sample frames"""
import sys
from pathlib import Path

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pickle
import numpy as np
import cv2
from juggle_buddy.ball_tracker import BallTracker

def main():
    print("=" * 60)
    print("Verifying Reference Path Extraction")
    print("=" * 60)
    
    # Load reference paths
    data_path = Path("data/reference_paths.pkl")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run extract_reference.py first.")
        return
    
    with open(data_path, 'rb') as f:
        paths = pickle.load(f)
    
    print(f"\n✅ Loaded {len(paths)} reference paths")
    
    # Show path statistics
    print("\nPath Statistics:")
    total_positions = 0
    for i, path in enumerate(paths):
        if len(path) > 0:
            total_positions += len(path)
            x_min, x_max = path[:, 0].min(), path[:, 0].max()
            y_min, y_max = path[:, 1].min(), path[:, 1].max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            print(f"  Ball {i}:")
            print(f"    Positions: {len(path)}")
            print(f"    X range: {x_min:.1f} to {x_max:.1f} (span: {x_range:.1f})")
            print(f"    Y range: {y_min:.1f} to {y_max:.1f} (span: {y_range:.1f})")
        else:
            print(f"  Ball {i}: EMPTY (no positions detected)")
    
    if total_positions == 0:
        print("\n⚠️  WARNING: No ball positions were extracted!")
        print("   This likely means ball detection failed.")
        print("   Check Step 3.1 to test ball detection.")
        return
    
    print(f"\nTotal positions extracted: {total_positions}")
    
    # Visualize detections on sample frames from video
    print("\n" + "=" * 60)
    print("Creating visualization of detected balls on sample frames...")
    print("=" * 60)
    
    video_path = Path("videos/cascade.mp4")
    if not video_path.exists():
        print(f"WARNING: Video {video_path} not found. Skipping frame visualization.")
        return
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames, {fps:.1f} FPS")
    
    # Process video again to show detections on frames
    tracker = BallTracker(num_balls=3)
    
    # Sample frames to visualize
    sample_frames = [0, total_frames // 4, total_frames // 2, 3 * total_frames // 4, total_frames - 1]
    sample_frames = [f for f in sample_frames if f < total_frames]
    
    output_dir = Path("data/verification")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = tracker.process_frame(frame, frame_num)
        
        # If this is a sample frame, save visualization
        if frame_num in sample_frames:
            vis_frame = frame.copy()
            
            # Get paths first to draw trail (past positions)
            paths = tracker.get_paths()
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR colors
            
            # Draw path trail (past positions) - light and subtle
            for i, path in enumerate(paths):
                if len(path) > 0 and frame_num < len(path.positions):
                    # Draw recent path trail (last 30 frames)
                    for j in range(max(0, frame_num - 30), frame_num):
                        if j < len(path.positions):
                            pt = tuple(path.positions[j].astype(int))
                            cv2.circle(vis_frame, pt, 1, colors[i], -1)
            
            # Draw detected balls ON TOP (current frame detections) - more prominent
            for ball in result['balls']:
                x, y = ball['x'], ball['y']
                r = ball.get('radius', 20)
                cv2.circle(vis_frame, (x, y), r, (0, 255, 0), 3)  # Green circle, thicker
                cv2.circle(vis_frame, (x, y), 3, (0, 0, 255), -1)  # Red center dot
            
            # Add text
            cv2.putText(vis_frame, f"Frame {frame_num} | Balls detected: {len(result['balls'])}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save frame
            output_path = output_dir / f"frame_{frame_num:05d}_detections.jpg"
            cv2.imwrite(str(output_path), vis_frame)
            print(f"  Saved visualization: {output_path}")
        
        frame_num += 1
    
    cap.release()
    
    print(f"\n✅ Verification complete!")
    print(f"   Check {output_dir} for sample frame visualizations")
    print(f"   These show detected balls (green circles) on frames from the video")

if __name__ == "__main__":
    main()
