# Test-Driven Development Guide

## Overview

This project uses a test-driven development (TDD) approach. Tests are organized to progressively build from simple components to complex integrations.

## Test Structure

### 1. **test_path_utils.py** - Foundation Tests
**Purpose**: Test the most basic data structures and utilities.

**What's Tested**:
- `BallPath` class creation and basic operations
- Path normalization (centering around origin)
- Path length calculations

**Why Start Here**: These are pure functions with no dependencies. Easy to implement and verify.

**Run Tests**:
```bash
pytest tests/test_path_utils.py -v
```

**Implementation Status**: ✅ Complete - These tests should pass with current implementation.

---

### 2. **test_ball_tracker.py** - Ball Detection Tests
**Purpose**: Test ball detection and tracking functionality.

**What's Tested**:
- Ball detection in images (using mock frames)
- BallTracker class initialization
- Frame processing
- Path accumulation over multiple frames

**Why Next**: Ball tracking is the foundation for everything else. Need to verify we can detect and track balls before building on top.

**Run Tests**:
```bash
pytest tests/test_ball_tracker.py -v
```

**Implementation Status**: ⚠️ Partial - Uses HoughCircles (placeholder). Will need YOLO for production.

**Next Steps**:
- Replace HoughCircles with YOLO detection
- Implement proper ball ID tracking across frames
- Add tests for occlusion handling

---

### 3. **test_reference_extractor.py** - Reference Path Extraction
**Purpose**: Test extracting ideal paths from reference video.

**What's Tested**:
- ReferenceExtractor initialization
- Extracting paths from video files
- Extracting paths from BallTracker
- Path normalization

**Why Next**: Once we can track balls, we need to extract reference paths.

**Run Tests**:
```bash
pytest tests/test_reference_extractor.py -v
```

**Implementation Status**: ✅ Complete - Basic functionality implemented.

**Next Steps**:
- Add cycle detection (identify complete juggling cycles)
- Add path smoothing/filtering
- Handle variable-length reference videos

---

### 4. **test_body_scaler.py** - Body Scaling Tests
**Purpose**: Test body keypoint detection and path scaling.

**What's Tested**:
- Body keypoint detection
- Scaling factor calculation
- Path scaling operations

**Why Next**: Need to scale reference to match live performer's body size.

**Run Tests**:
```bash
pytest tests/test_body_scaler.py -v
```

**Implementation Status**: ⚠️ Partial - Uses placeholder keypoint detection. Needs real pose estimation.

**Next Steps**:
- Integrate OpenCV DNN pose estimation
- Add MediaPipe as optional fallback
- Test with real video frames

---

### 5. **test_path_comparator.py** - Path Comparison & Scoring
**Purpose**: Test comparing live paths to reference and computing scores.

**What's Tested**:
- Path alignment (temporal)
- Deviation calculation (horizontal, vertical, total)
- Score computation (1-100 range)
- PathComparator class

**Why Next**: This is the core functionality - comparing performance to ideal.

**Run Tests**:
```bash
pytest tests/test_path_comparator.py -v
```

**Implementation Status**: ✅ Complete - Basic scoring implemented.

**Next Steps**:
- Implement dynamic time warping for better alignment
- Tune scoring metrics based on real data
- Add cycle-based scoring (every 90 frames)

---

### 6. **test_integration.py** - Integration Tests
**Purpose**: Test that modules work together correctly.

**What's Tested**:
- Tracker → Extractor flow
- Scaler → Comparator flow
- End-to-end workflow

**Why Last**: Need all individual components working before testing integration.

**Run Tests**:
```bash
pytest tests/test_integration.py -v
```

**Implementation Status**: ✅ Complete - Basic integration tests.

**Next Steps**:
- Add tests with real video files
- Test error handling across modules
- Performance tests

---

## Running All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=juggle_buddy --cov-report=html

# Run specific test file
pytest tests/test_path_utils.py -v

# Run specific test
pytest tests/test_path_utils.py::TestBallPath::test_ball_path_creation -v
```

## Progressive Development Workflow

### Step 1: Run Tests (They Should Fail Initially)
```bash
pytest tests/test_path_utils.py -v
```

### Step 2: Implement Minimal Code to Pass Tests
Write just enough code in `juggle_buddy/path_utils.py` to make tests pass.

### Step 3: Refactor
Once tests pass, refactor for better code quality while keeping tests green.

### Step 4: Move to Next Module
Repeat for next test file in sequence.

## Test Coverage Goals

- **Unit Tests**: 80%+ coverage for each module
- **Integration Tests**: Cover all major workflows
- **Edge Cases**: Empty paths, missing detections, etc.

## Adding New Tests

When adding new functionality:

1. **Write test first** (TDD approach)
2. **Run test** (should fail)
3. **Implement feature** (make test pass)
4. **Refactor** (keep tests passing)

Example:
```python
def test_new_feature():
    """Test description."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = new_function(input_data)
    
    # Assert
    assert result == expected_output
```

## Mock Data for Testing

Tests use synthetic data to avoid requiring actual video files:

```python
# Create test image
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.circle(img, (100, 100), 20, (255, 255, 255), -1)

# Create test paths
path = np.array([[0, 0], [10, 10], [20, 20]])
```

## Common Test Patterns

### Testing with Mock Objects
```python
class MockTracker:
    def get_paths(self):
        return [BallPath(0, np.array([[0, 0]]))]
```

### Testing Edge Cases
```python
def test_empty_path():
    path = np.array([]).reshape(0, 2)
    result = process_path(path)
    assert result is not None
```

### Testing Error Handling
```python
def test_invalid_input():
    with pytest.raises(ValueError):
        function_with_validation(None)
```

## Next Test Files to Create

1. **test_video_recorder.py** - Video I/O utilities
2. **test_cycle_detector.py** - Cycle detection for scoring every 90 frames
3. **test_visualization.py** - Visualization/feedback utilities
4. **test_main.py** - End-to-end application tests

## Debugging Failed Tests

```bash
# Run with verbose output
pytest tests/test_path_utils.py -v -s

# Run with print statements visible
pytest tests/test_path_utils.py -v -s --capture=no

# Run with debugger
pytest tests/test_path_utils.py --pdb
```

## Continuous Integration

Consider adding:
- GitHub Actions for automated testing
- Pre-commit hooks to run tests
- Coverage reporting
