# Quick Start Guide

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify installation**:
```bash
pytest tests/test_path_utils.py -v
```

## Project Structure

```
juggle_buddy/
├── juggle_buddy/          # Main package
│   ├── __init__.py
│   ├── path_utils.py      # Path data structures
│   ├── ball_tracker.py    # Ball detection & tracking
│   ├── reference_extractor.py  # Extract ideal paths
│   ├── body_scaler.py     # Body scaling
│   └── path_comparator.py # Path comparison & scoring
├── tests/                 # Test suite
│   ├── test_path_utils.py
│   ├── test_ball_tracker.py
│   ├── test_reference_extractor.py
│   ├── test_body_scaler.py
│   ├── test_path_comparator.py
│   └── test_integration.py
├── requirements.txt
├── pytest.ini
└── README.md
```

## Development Workflow

### 1. Start with Tests
Run tests to see current status:
```bash
pytest tests/ -v
```

### 2. Implement Features
Follow the test-driven approach:
- Start with `test_path_utils.py` (simplest)
- Move to `test_ball_tracker.py`
- Continue through integration tests

### 3. Run Tests Frequently
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=juggle_buddy --cov-report=html
```

## Next Steps

1. **Implement Video Recording**:
   - Create `video_recorder.py` module
   - Add tests in `test_video_recorder.py`

2. **Improve Ball Tracking**:
   - Integrate YOLO for detection
   - Implement proper ID tracking

3. **Add Body Detection**:
   - Integrate OpenCV DNN pose estimation
   - Test with real video frames

4. **Create Main Application**:
   - Build `main.py` to orchestrate modules
   - Add user interface/feedback

## Key Files to Read

1. **README.md** - Project overview and architecture
2. **DESIGN_SUGGESTIONS.md** - Detailed design recommendations
3. **TESTING_GUIDE.md** - Complete testing strategy

## Tips

- Tests are designed to be run in order (simplest to most complex)
- Each module can be developed independently
- Use mock data in tests to avoid requiring video files initially
- Start with unit tests, then move to integration tests
