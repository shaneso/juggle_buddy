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

### Step 2.1: Extract Reference Paths

✅ **Script already exists**: `scripts/extract_reference.py`

**Action**: Run the extraction script:
```bash
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

✅ **Script already exists**: `scripts/visualize_reference.py`

**Action**: Run the visualization script:
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

✅ **Script already exists**: `scripts/test_ball_detection.py`

**Action**: Run the detection test script:
```bash
python scripts/test_ball_detection.py
```

**Expected**: Images saved showing detected balls (if any).

**If detection is poor**: This is expected. We'll improve it next.

---

### Step 3.2: Ball Detection - Calibrate Colors (if needed)

✅ **DONE**: Both YOLO and multi-color detection are implemented in `juggle_buddy/ball_tracker.py`.

**Current Implementation**:
- Default: Multi-color detection (red, blue, green) - most reliable for specific ball colors
- YOLO also available as option

**If ball detection is poor**: Use the calibration tool to find HSV ranges for your balls:

```bash
python scripts/calibrate_ball_colors.py
```

This interactive tool helps you find the right HSV color ranges by adjusting sliders in real-time.

**See**: `juggle_buddy/ball_tracker.py` for implementation details.

---

## Phase 4: Body Scaling

### Step 4.1: Extract Body Keypoints from Reference Video

✅ **Script already exists**: `scripts/extract_reference_keypoints.py`

✅ **Implementation**: Uses YOLO v8 pose estimation (nano model) in `juggle_buddy/body_scaler.py`.

**Action**: Run the keypoint extraction script:
```bash
python scripts/extract_reference_keypoints.py
```

**Expected**: Keypoints extracted and saved (may be approximate with current implementation).

---

## Phase 5: Live Tracking Setup

### Step 5.1: Test Live Tracking

✅ **Script already exists**: `scripts/test_live_tracking.py`

**Action**: Run the live tracking test:
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

✅ **Script already exists**: `scripts/test_path_comparison.py`

**Action**: Run the path comparison test:
```bash
python scripts/test_path_comparison.py
```

**Expected**: Scores showing perfect match = 100, deviations = lower scores.

---

## Phase 7: Complete Integration

### Step 7.1: Run Main Application

✅ **Main application already exists**: `main.py`

**Action**: Run the main application:
```bash
python main.py
```

**Controls**:
- `q` - Quit
- `r` - Reset tracking
- `s` - Show current score

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
