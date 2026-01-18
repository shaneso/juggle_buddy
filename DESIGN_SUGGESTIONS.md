# Design Suggestions for Juggle Buddy

## Architecture Overview

The system is designed with a modular, test-driven approach. Each module has clear responsibilities and can be developed and tested independently.

## Key Design Decisions & Suggestions

### 1. **Ball Tracking Strategy**

**Current Approach**: HoughCircles for initial detection (simple, works for testing)
**Production Recommendation**: 
- Use YOLO v8 (ultralytics) for robust ball detection
- Use OpenCV's CSRT or KCF trackers for frame-to-frame tracking
- Implement Hungarian algorithm for ball ID assignment across frames
- Consider color-based tracking as a fallback if YOLO fails

**Why**: HoughCircles is sensitive to lighting and background. YOLO provides more robust detection, and dedicated trackers handle occlusion better.

### 2. **Body Scaling Approach**

**Current Approach**: Simple keypoint detection with placeholders
**Production Recommendation**:
- **Primary**: Use OpenCV's DNN pose estimation (OpenPose or similar) - more stable across Python versions
- **Fallback**: MediaPipe if available (better accuracy but version-dependent)
- Use multiple keypoints (shoulders, hips, head) for more robust scaling
- Calculate scaling factor as weighted average of width and height ratios

**Why**: Body size varies, so scaling is critical for accurate comparison. Using multiple keypoints reduces error from single-point measurements.

### 3. **Path Representation**

**Current Approach**: NumPy arrays of (x, y) coordinates
**Suggestion**: 
- Consider adding timestamps for temporal analysis
- Store paths with frame numbers for cycle detection
- Normalize paths early to reduce computational overhead

**Why**: Temporal information helps with cycle alignment and pattern recognition.

### 4. **Scoring Metric**

**Current Approach**: Simple mean deviation normalized to 1-100
**Suggestions for Improvement**:
- **Horizontal vs Vertical Weighting**: Vertical deviations might be more critical for juggling
- **Cycle-based Scoring**: Score every 90 frames (as specified) by comparing complete cycles
- **Velocity Matching**: Consider comparing ball velocities, not just positions
- **Pattern Recognition**: Detect if user is doing a different pattern entirely (score = 0)

**Scoring Formula Options**:
```python
# Option 1: Weighted deviation
score = 100 - (w_h * horizontal_dev + w_v * vertical_dev) * 100

# Option 2: Cycle-based (compare complete cycles)
score = 100 - cycle_deviation * normalization_factor

# Option 3: Multi-metric (position + velocity + timing)
score = weighted_average(position_score, velocity_score, timing_score)
```

### 5. **Temporal Alignment**

**Current Approach**: Simple padding/truncation
**Production Recommendation**:
- Use cross-correlation to find optimal temporal offset
- Implement dynamic time warping (DTW) for non-linear alignment
- Detect cycle boundaries automatically
- Handle variable-speed juggling

**Why**: Jugglers don't maintain perfect rhythm. DTW can align paths even when timing varies.

### 6. **Performance Optimizations**

**Suggestions**:
- Process frames in batches rather than one-by-one
- Use GPU acceleration for YOLO detection
- Cache reference paths after normalization
- Only recalculate scaling when body position changes significantly
- Downsample video for faster processing (if accuracy allows)

### 7. **Error Handling & Robustness**

**Critical Areas**:
- Handle missing ball detections (interpolate or skip)
- Handle occlusions (predict ball position)
- Handle different lighting conditions
- Handle different camera angles
- Validate that 3 balls are actually being tracked

### 8. **User Experience**

**Suggestions**:
- Visual feedback: Overlay ideal paths on live video
- Real-time score display
- Highlight which ball is deviating most
- Provide actionable feedback ("Ball 2 is too high", "Timing is off")
- Save performance videos for review

### 9. **Testing Strategy**

**Progressive Development**:
1. ✅ Unit tests for data structures (path_utils)
2. ✅ Unit tests for ball tracking (with mock frames)
3. ✅ Unit tests for reference extraction
4. ✅ Unit tests for body scaling
5. ✅ Unit tests for path comparison
6. ✅ Integration tests
7. **Next**: End-to-end tests with sample videos
8. **Next**: Performance tests with real juggling videos

### 10. **Extension Points for Future Patterns**

**Design for Extensibility**:
- Abstract pattern class (CascadePattern, ShowerPattern, etc.)
- Pattern-specific reference extractors
- Pattern-specific scoring metrics
- Configuration files for pattern parameters

## Implementation Priority

### Phase 1: Core Functionality (Current)
- [x] Basic path data structures
- [x] Ball detection (simple)
- [x] Reference extraction
- [x] Path comparison
- [ ] Video recording utilities

### Phase 2: Production Quality
- [ ] YOLO integration for ball detection
- [ ] Robust ball tracking with ID management
- [ ] Real body keypoint detection
- [ ] Cycle detection and alignment
- [ ] Improved scoring metrics

### Phase 3: Polish
- [ ] Real-time visualization
- [ ] User interface
- [ ] Performance optimizations
- [ ] Error handling and edge cases

### Phase 4: Extensions
- [ ] Support for other 3-ball patterns
- [ ] Multi-pattern comparison
- [ ] Performance analytics over time
- [ ] Export/import reference patterns

## Technical Debt & Considerations

1. **Ball Detection**: Current HoughCircles is a placeholder - needs YOLO
2. **Body Detection**: Placeholder implementation - needs real pose estimation
3. **Scoring Tuning**: Max deviation (100 pixels) is a heuristic - needs calibration
4. **Temporal Alignment**: Simple approach - DTW would be better
5. **No Video I/O**: Need to add video recording/playback utilities

## Questions to Consider

1. **Frame Rate**: What FPS are you targeting? Affects tracking accuracy.
2. **Video Resolution**: Higher res = better tracking but slower processing
3. **Real-time vs Post-processing**: Real-time is harder but better UX
4. **Reference Video Quality**: How will you ensure reference is "ideal"?
5. **Scoring Calibration**: How will you validate that scores are meaningful?
