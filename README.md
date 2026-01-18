# Juggle Buddy - 3-Ball Cascade Tracking System

## Project Overview
A system that tracks ball paths during juggling and compares them against an ideal reference pattern, providing real-time feedback scores.

## Architecture Suggestions

### Recommended Module Structure

1. **`video_recorder.py`** - Handles video capture from webcam
   - Record reference video
   - Record live performance
   - Frame extraction utilities

2. **`ball_tracker.py`** - Ball detection and tracking
   - Uses YOLO for ball detection
   - Tracks 3 balls individually across frames
   - Returns ball positions (x, y, frame_number) for each ball

3. **`reference_extractor.py`** - Creates ideal path from reference video
   - Processes reference video to extract ball paths
   - Normalizes paths (centers around juggler)
   - Stores reference paths as numpy arrays

4. **`body_scaler.py`** - Scales reference to match live performer
   - Detects body keypoints (shoulders, hips) using OpenCV or MediaPipe
   - Calculates scaling factor based on body dimensions
   - Applies scaling to reference paths

5. **`path_comparator.py`** - Compares live paths to reference
   - Aligns live paths with reference paths (temporal alignment)
   - Calculates horizontal and vertical deviations
   - Computes score (1-100) based on deviation metrics

6. **`main.py`** - Main application controller
   - Orchestrates all modules
   - Handles user interface/feedback
   - Manages video recording workflow

### Key Design Decisions

1. **Ball Tracking**: Use YOLO v8 (ultralytics) for detection, then use OpenCV's tracking algorithms (CSRT, KCF) for frame-to-frame tracking
2. **Body Scaling**: Use OpenCV's DNN pose estimation as primary (more stable across Python versions), MediaPipe as optional enhancement
3. **Path Representation**: Store paths as numpy arrays of (x, y, frame) tuples, one array per ball
4. **Scoring Metric**: 
   - Horizontal deviation: distance from ideal x position
   - Vertical deviation: distance from ideal y position
   - Temporal alignment: match cycles using cross-correlation
   - Score = 100 - (normalized_deviation * 100), clamped to 1-100

### Progressive Development Strategy

Start with unit tests for each module, building from simplest to most complex:
1. Path data structures and utilities
2. Ball tracking (with mock frames)
3. Reference extraction
4. Body scaling
5. Path comparison
6. Integration tests

## Setup

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=. --cov-report=html
```

## Usage

```bash
# Record reference video
python main.py --record-reference

# Run live tracking
python main.py --live
```
