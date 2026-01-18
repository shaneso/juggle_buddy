# Juggle Buddy - 3-Ball Cascade Tracking System

## Project Overview
A system that tracks ball paths during juggling and compares them against an ideal reference pattern, providing real-time feedback scores (1-100) based on deviation from the reference path.

## Current Status

✅ **Core Modules Implemented**:
- `path_utils.py` - Ball path data structures
- `ball_tracker.py` - Ball detection and tracking (HoughCircles)
- `reference_extractor.py` - Extract ideal paths from reference video
- `body_scaler.py` - Body scaling for different performer sizes
- `path_comparator.py` - Path comparison and scoring (1-100)
- `video_recorder.py` - Video I/O utilities

✅ **Scripts Available**:
- `scripts/extract_reference.py` - Extract reference paths from `videos/cascade.mp4`
- `scripts/verify_extraction.py` - Verify extracted paths with visual output
- `scripts/visualize_reference.py` - Plot reference paths
- `scripts/test_ball_detection.py` - Test ball detection on sample frames
- `scripts/extract_reference_keypoints.py` - Extract body keypoints
- `scripts/test_live_tracking.py` - Test live webcam tracking
- `scripts/test_path_comparison.py` - Test path comparison logic
- `main.py` - Main application (live tracking with scoring)

✅ **Test Suite**: Full unit and integration tests in `tests/`

## Architecture

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

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify reference video exists
python -c "from pathlib import Path; print('Video exists:', Path('videos/cascade.mp4').exists())"
```

### 2. Extract Reference Paths
```bash
# Extract ball paths from reference video
python scripts/extract_reference.py

# Verify extraction (optional but recommended)
python scripts/verify_extraction.py

# Visualize paths (optional)
python scripts/visualize_reference.py
```

### 3. Run Live Tracking
```bash
# Start main application
python main.py
```

**Controls**:
- `q` - Quit
- `r` - Reset tracking
- `s` - Show current score

## Usage

### Extract Reference Data
```bash
# Extract reference paths (uses videos/cascade.mp4)
python scripts/extract_reference.py

# Extract body keypoints for scaling
python scripts/extract_reference_keypoints.py
```

### Testing Individual Components
```bash
# Test ball detection
python scripts/test_ball_detection.py

# Test live tracking
python scripts/test_live_tracking.py

# Test path comparison
python scripts/test_path_comparison.py
```

### Run Main Application
```bash
# Start live tracking with scoring
python main.py
```

## Project Structure

```
juggle_buddy/
├── juggle_buddy/          # Core package modules
│   ├── path_utils.py      # Ball path data structures
│   ├── ball_tracker.py    # Ball detection & tracking
│   ├── reference_extractor.py  # Extract ideal paths
│   ├── body_scaler.py     # Body scaling
│   ├── path_comparator.py # Path comparison & scoring
│   └── video_recorder.py  # Video I/O utilities
├── scripts/               # Utility scripts
│   ├── extract_reference.py
│   ├── verify_extraction.py
│   ├── visualize_reference.py
│   ├── test_ball_detection.py
│   ├── extract_reference_keypoints.py
│   ├── test_live_tracking.py
│   └── test_path_comparison.py
├── tests/                 # Test suite
├── videos/                # Video files
│   └── cascade.mp4        # Reference video
├── data/                  # Generated data (created at runtime)
│   ├── reference_paths.pkl
│   └── reference_keypoints.pkl
└── main.py               # Main application
```

## Documentation

- **`STEP_BY_STEP_GUIDE.md`** - Detailed implementation guide (follow this!)
- **`DESIGN_SUGGESTIONS.md`** - Architecture and design recommendations
- **`TESTING_GUIDE.md`** - Testing strategy
- **`QUICK_REFERENCE.md`** - Quick command reference
- **`TROUBLESHOOTING.md`** - Common issues and solutions
