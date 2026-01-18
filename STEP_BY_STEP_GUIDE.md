# Step-by-Step Implementation Guide

This guide breaks down the implementation into detailed, actionable steps. Follow them in order.

## Prerequisites Check

### Step 0.1: Verify Setup
```bash
# Check Python version (should be 3.8+)
python --version

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2, numpy, pytest; print('All imports successful')"
```

### Step 0.2: Verify Reference Video
```bash
# Check that reference video exists
python -c "from pathlib import Path; print('Video exists:', Path('videos/cascade.mp4').exists())"
```

---

## Phase 1: Foundation - Path Utilities (COMPLETE ✅)

### Step 1.1: Test Path Utilities
```bash
pytest tests/test_path_utils.py -v
```
**Expected**: All tests should pass ✅

**What this verifies**: Basic data structures work correctly.

**If tests fail**: Check that `juggle_buddy/path_utils.py` is implemented correctly.

---

## Phase 2: Extract Reference Path from Video

### Step 2.1: Create Reference Extraction Script

Create a script to extract the reference path from your video:

**File**: `scripts/extract_reference.py`

```python
"""Script to extract reference ball paths from cascade.mp4"""
from juggle_buddy.reference_extractor import ReferenceExtractor
from juggle_buddy.reference_extractor import normalize_reference
import numpy as np
import pickle
from pathlib import Path

def main():
    print("=" * 60)
    print("Extracting Reference Ball Paths from cascade.mp4")
    print("=" * 60)
    
    # Create extractor
    extractor = ReferenceExtractor(num_balls=3)
    
    # Extract paths (will use videos/cascade.mp4 by default)
    paths = extractor.extract_from_video()
    
    if paths is None or len(paths) == 0:
        print("ERROR: Failed to extract paths from video")
        return
    
    print(f"\nExtracted {len(paths)} ball paths")
    for i, path in enumerate(paths):
        print(f"  Ball {i}: {len(path)} positions")
    
    # Normalize paths
    normalized = normalize_reference(paths)
    
    # Save reference paths
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "reference_paths.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(normalized, f)
    
    print(f"\n✅ Reference paths saved to: {output_path}")
    print(f"   Total positions: {sum(len(p) for p in normalized)}")
    
    # Display statistics
    print("\nPath Statistics:")
    for i, path in enumerate(normalized):
        if len(path) > 0:
            x_range = path[:, 0].max() - path[:, 0].min()
            y_range = path[:, 1].max() - path[:, 1].min()
            print(f"  Ball {i}: X range: {x_range:.1f}, Y range: {y_range:.1f}")

if __name__ == "__main__":
    main()
```

**Action**: Create this file and run it:
```bash
mkdir scripts
# (Create the file above)
python scripts/extract_reference.py
```

**Expected Output**: 
- Should process the video
- Extract 3 ball paths
- Save to `data/reference_paths.pkl`
- Display statistics

**Verify Output**: After running, verify the extraction worked:
```bash
python scripts/verify_extraction.py
```
This will:
- Show detailed statistics about extracted paths
- Create sample frame images showing detected balls (saved to `data/verification/`)
- Help you see if balls were actually detected or if detection failed

**Troubleshooting**:
- If ball detection fails: The HoughCircles method may not work well with your video. This is expected - we'll improve it in Step 3.
- If paths are empty: Check video quality, lighting, ball visibility.
- **Quick visual check**: Run `verify_extraction.py` to see sample frames with detected balls overlaid.

---

### Step 2.2: Visualize Reference Paths

Create a visualization script to see what was extracted:

**File**: `scripts/visualize_reference.py`

```python
"""Visualize extracted reference paths"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Load reference paths
    data_path = Path("data/reference_paths.pkl")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run extract_reference.py first.")
        return
    
    with open(data_path, 'rb') as f:
        paths = pickle.load(f)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green']
    for i, path in enumerate(paths):
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], 
                   color=colors[i], label=f'Ball {i}', 
                   linewidth=2, alpha=0.7)
            ax.scatter(path[0, 0], path[0, 1], 
                      color=colors[i], s=100, marker='o', 
                      label=f'Ball {i} start')
            ax.scatter(path[-1, 0], path[-1, 1], 
                      color=colors[i], s=100, marker='s', 
                      label=f'Ball {i} end')
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('Reference Ball Paths (Normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Save plot
    output_path = Path("data/reference_paths_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
```

**Action**: Create and run:
```bash
python scripts/visualize_reference.py
```

**Expected**: A plot showing the 3 ball paths overlaid.

**What to look for**:
- Are there 3 distinct paths?
- Do they form a cascade pattern (figure-8 shape)?
- Are paths smooth or noisy?

---

## Phase 3: Improve Ball Detection

### Step 3.1: Test Current Ball Detection

The current implementation uses HoughCircles which may not work well. Let's test it:

**File**: `scripts/test_ball_detection.py`

```python
"""Test ball detection on a single frame from reference video"""
import cv2
import numpy as np
from juggle_buddy.ball_tracker import detect_balls
from pathlib import Path

def main():
    video_path = Path("videos/cascade.mp4")
    cap = cv2.VideoCapture(str(video_path))
    
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
            x, y, r = ball['x'], ball['y'], ball['radius']
            cv2.circle(vis_frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis_frame, (x, y), 2, (0, 0, 255), -1)
        
        # Save visualization
        output_path = f"data/detection_test_frame_{frame_num}.jpg"
        cv2.imwrite(output_path, vis_frame)
        print(f"  Saved to: {output_path}")
    
    cap.release()

if __name__ == "__main__":
    main()
```

**Action**: Create and run:
```bash
python scripts/test_ball_detection.py
```

**Expected**: Images saved showing detected balls (if any).

**If detection is poor**: This is expected. We'll improve it next.

---

### Step 3.2: Improve Ball Detection (YOLO Integration)

✅ **DONE**: Color-based detection has been implemented by replacing `detect_balls()` in `ball_tracker.py`.

The function now uses HSV color range masking instead of HoughCircles. The default detects bright/white objects, but you can adjust the color range.

**Current Implementation**: `detect_balls()` now uses color-based detection with optional `color_range` parameter.

**To adjust for your ball colors**: Modify the HSV values in `juggle_buddy/ball_tracker.py` lines 28-29, or pass a custom `color_range` when calling the function.

---

**Option A: Color-based Detection** (✅ Already Implemented)

The `detect_balls()` function now uses color-based detection. For reference, here's what was implemented:

```python
def detect_balls_by_color(img: np.ndarray, color_range: tuple = None) -> List[Dict]:
    """
    Detect balls by color (useful if balls are distinct colors).
    
    Args:
        img: Input image (BGR)
        color_range: HSV color range tuple ((lower_h, lower_s, lower_v), (upper_h, upper_s, upper_v))
        
    Returns:
        List of detected balls
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Default: detect bright/white objects (adjust for your balls)
    if color_range is None:
        lower = np.array([0, 0, 200])  # Adjust these values
        upper = np.array([180, 30, 255])
    else:
        lower, upper = color_range
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    balls = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 5000:  # Filter by size
            (x, y), radius = cv2.minEnclosingCircle(contour)
            balls.append({
                'x': int(x),
                'y': int(y),
                'radius': int(radius)
            })
    
    return balls
```

**Option B: YOLO Integration (Better, but more setup)**

1. Download YOLO model or use ultralytics:
```python
from ultralytics import YOLO

def detect_balls_yolo(img: np.ndarray, model_path: str = None) -> List[Dict]:
    """Detect balls using YOLO."""
    if model_path is None:
        # Use pre-trained COCO model (detects "sports ball")
        model = YOLO('yolov8n.pt')
    else:
        model = YOLO(model_path)
    
    results = model(img, classes=[32])  # Class 32 = sports ball
    
    balls = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            radius = int((x2 - x1) / 2)
            balls.append({'x': x, 'y': y, 'radius': radius})
    
    return balls
```

**Action**: Choose one approach and integrate it into `ball_tracker.py`.

**Testing**: Re-run `scripts/test_ball_detection.py` to see improved results.

---

## Phase 4: Body Scaling

### Step 4.1: Extract Body Keypoints from Reference Video

Create a script to extract body keypoints from the reference video:

**File**: `scripts/extract_reference_keypoints.py`

```python
"""Extract body keypoints from reference video"""
import cv2
import numpy as np
from juggle_buddy.body_scaler import detect_body_keypoints
from pathlib import Path
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
```

**Action**: Create and run:
```bash
python scripts/extract_reference_keypoints.py
```

**Expected**: Keypoints extracted and saved (may be approximate with current implementation).

---

## Phase 5: Live Tracking Setup

### Step 5.1: Create Live Tracking Script

Create a script to test live ball tracking:

**File**: `scripts/test_live_tracking.py`

```python
"""Test live ball tracking from webcam"""
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
            x, y, r = ball['x'], ball['y'], ball.get('radius', 20)
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
```

**Action**: Create and run:
```bash
python scripts/test_live_tracking.py
```

**Expected**: Webcam window opens, shows ball detections in real-time.

**Testing**: 
- Wave objects in front of camera
- Check if balls are detected
- Press 's' to see current paths

---

## Phase 6: Path Comparison

### Step 6.1: Test Path Comparison

Create a script to test comparing paths:

**File**: `scripts/test_path_comparison.py`

```python
"""Test path comparison with reference paths"""
import pickle
import numpy as np
from pathlib import Path
from juggle_buddy.path_comparator import PathComparator

def main():
    # Load reference paths
    ref_path = Path("data/reference_paths.pkl")
    if not ref_path.exists():
        print(f"ERROR: {ref_path} not found. Run extract_reference.py first.")
        return
    
    with open(ref_path, 'rb') as f:
        reference_paths = pickle.load(f)
    
    print(f"Loaded {len(reference_paths)} reference paths")
    
    # Create comparator
    comparator = PathComparator()
    comparator.set_reference(reference_paths)
    
    # Test 1: Perfect match
    print("\nTest 1: Perfect match")
    perfect_paths = [path.copy() for path in reference_paths]
    scores = comparator.compare(perfect_paths)
    print(f"  Overall score: {scores['overall_score']}")
    print(f"  Ball scores: {scores['ball_scores']}")
    
    # Test 2: Slight deviation
    print("\nTest 2: Slight deviation (+10 pixels)")
    deviated_paths = [path + 10 for path in reference_paths]
    scores = comparator.compare(deviated_paths)
    print(f"  Overall score: {scores['overall_score']}")
    print(f"  Ball scores: {scores['ball_scores']}")
    
    # Test 3: Large deviation
    print("\nTest 3: Large deviation (+50 pixels)")
    large_dev_paths = [path + 50 for path in reference_paths]
    scores = comparator.compare(large_dev_paths)
    print(f"  Overall score: {scores['overall_score']}")
    print(f"  Ball scores: {scores['ball_scores']}")

if __name__ == "__main__":
    main()
```

**Action**: Create and run:
```bash
python scripts/test_path_comparison.py
```

**Expected**: Scores showing perfect match = 100, deviations = lower scores.

---

## Phase 7: Complete Integration

### Step 7.1: Create Main Application

Create the main application that ties everything together:

**File**: `main.py`

```python
"""Main application for Juggle Buddy"""
import cv2
import pickle
import numpy as np
from pathlib import Path
from juggle_buddy.ball_tracker import BallTracker
from juggle_buddy.reference_extractor import ReferenceExtractor, normalize_reference
from juggle_buddy.body_scaler import BodyScaler, detect_body_keypoints
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
            x, y, r = ball['x'], ball['y'], ball.get('radius', 20)
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
```

**Action**: Create this file. Note: You'll need to add missing imports:
```python
from juggle_buddy.body_scaler import calculate_scaling_factor, scale_paths
```

**Run**:
```bash
python main.py
```

---

## Summary Checklist

Follow these steps in order:

- [ ] **Step 0.1-0.2**: Verify setup and reference video
- [ ] **Step 1.1**: Test path utilities (should pass)
- [ ] **Step 2.1**: Extract reference paths from video
- [ ] **Step 2.2**: Visualize reference paths
- [ ] **Step 3.1**: Test current ball detection
- [ ] **Step 3.2**: Improve ball detection (YOLO or color-based)
- [ ] **Step 4.1**: Extract body keypoints from reference
- [ ] **Step 5.1**: Test live tracking
- [ ] **Step 6.1**: Test path comparison
- [ ] **Step 7.1**: Create and run main application

Each step builds on the previous one. Complete them sequentially for best results.
