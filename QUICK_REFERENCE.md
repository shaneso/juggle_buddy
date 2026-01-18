# Quick Reference Card

## Immediate Next Steps (In Order)

### 1. Extract Reference Paths (START HERE)
```bash
python scripts/extract_reference.py
```
**What it does**: Processes `videos/cascade.mp4` and extracts ball paths
**Output**: `data/reference_paths.pkl`
**Time**: ~1-2 minutes depending on video length

### 2. Visualize Reference Paths (Optional but Recommended)
```bash
# First install matplotlib if needed
pip install matplotlib

# Then create and run visualization script (see STEP_BY_STEP_GUIDE.md Step 2.2)
python scripts/visualize_reference.py
```
**What it does**: Shows a plot of the extracted ball paths
**Output**: `data/reference_paths_visualization.png`

### 3. Test Ball Detection
```bash
# Create test script (see STEP_BY_STEP_GUIDE.md Step 3.1)
python scripts/test_ball_detection.py
```
**What it does**: Tests if balls can be detected in video frames
**Output**: Images showing detected balls

### 4. Test Live Tracking
```bash
# Create test script (see STEP_BY_STEP_GUIDE.md Step 5.1)
python scripts/test_live_tracking.py
```
**What it does**: Opens webcam and tracks balls in real-time
**Controls**: 'q' to quit, 's' to show paths

### 5. Run Main Application
```bash
python main.py
```
**What it does**: Complete system - tracks live juggling and compares to reference
**Controls**: 'q' to quit, 'r' to reset, 's' to show score

## File Structure

```
juggle_buddy/
├── videos/
│   └── cascade.mp4          ← Your reference video (already here)
├── juggle_buddy/            ← Core modules
├── scripts/                 ← Utility scripts
│   └── extract_reference.py ← Run this first!
├── tests/                   ← Unit tests
├── data/                    ← Generated data (created when you run scripts)
│   ├── reference_paths.pkl
│   └── reference_keypoints.pkl
└── main.py                  ← Main application
```

## Common Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_path_utils.py -v

# Extract reference (do this first!)
python scripts/extract_reference.py

# Run main app
python main.py
```

## Troubleshooting

**"Could not open video file"**
- Check that `videos/cascade.mp4` exists
- Verify file path is correct

**"No balls detected"**
- Current detection uses HoughCircles (basic)
- May need to improve detection (see Step 3.2 in guide)
- Check video quality/lighting

**"Import errors"**
- Run: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

## Key Files to Read

1. **STEP_BY_STEP_GUIDE.md** - Detailed implementation steps
2. **DESIGN_SUGGESTIONS.md** - Architecture and design decisions
3. **TESTING_GUIDE.md** - Testing strategy

## Progress Tracking

- [ ] Step 0: Setup verified
- [ ] Step 1: Path utilities tested
- [ ] Step 2: Reference paths extracted
- [ ] Step 3: Ball detection improved
- [ ] Step 4: Body keypoints extracted
- [ ] Step 5: Live tracking working
- [ ] Step 6: Path comparison tested
- [ ] Step 7: Main app running
