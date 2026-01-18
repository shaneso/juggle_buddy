# Juggle Buddy - 3-Ball Cascade Tracking System

## Project Overview
A system that tracks ball paths during juggling and compares them against an ideal reference pattern, providing real-time feedback scores (1-100) based on deviation from the reference path.

## Current Status

✅ **Core Modules Implemented**:
- `path_utils.py` - Ball path data structures
- `ball_tracker.py` - Ball detection and tracking (YOLO v8 + color-based fallback)
- `reference_extractor.py` - Extract ideal paths from reference video
- `body_scaler.py` - Body scaling using YOLO v8 pose estimation
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

All core modules are implemented. See `DESIGN_SUGGESTIONS.md` for detailed architecture information.

### Key Design Decisions

1. **Ball Tracking**: Multi-color detection (red, blue, green) by default - more reliable than YOLO for specific ball colors. YOLO available as option.
2. **Body Scaling**: Uses YOLO v8 pose estimation (nano model) for body keypoint detection
3. **Path Representation**: NumPy arrays of (x, y) coordinates, one array per ball
4. **Scoring Metric**: Deviation-based scoring (1-100) comparing live paths to normalized reference paths

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

## Ball Color Calibration

If ball detection is poor, calibrate your ball colors:

```bash
# Run calibration tool to find HSV ranges for your balls
python scripts/calibrate_ball_colors.py
```

This interactive tool helps you:
- Find HSV color ranges for red, blue, and green balls
- See real-time preview of detected colors
- Save calibrated values to use in ball detection

**Note**: By default, ball detection uses multi-color detection (red, blue, green) which is more reliable than YOLO for specific colored balls.

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
