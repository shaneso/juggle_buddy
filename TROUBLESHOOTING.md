# Troubleshooting Guide

## Dependency Conflicts

### Issue: gpy (GPy) dependency conflicts

**Error Message**:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. 
This behaviour is the source of the following dependency conflicts.
gpy 1.13.2 requires numpy<2.0.0,>=1.7, but you have numpy 2.2.6 which is incompatible.
gpy 1.13.2 requires scipy<=1.12.0,>=1.3.0, but you have scipy 1.17.0 which is incompatible.
```

**Solution**: This project doesn't use `gpy`. The requirements.txt has been updated to use compatible versions:
- numpy: `>=1.24.0,<2.0.0`
- scipy: `>=1.11.0,<=1.12.0`

**What to do**:
1. Reinstall requirements: `pip install -r requirements.txt --upgrade`
2. If you need `gpy` for another project, consider using separate virtual environments

**Alternative**: If you don't need `gpy` at all, you can ignore this warning. The juggling project will work fine with the updated requirements.

---

## Installation Issues

### Issue: "pip: command not found"

**Solution**: 
- Windows: Try `py -m pip install -r requirements.txt`
- Or install pip: `python -m ensurepip --upgrade`

### Issue: "python: command not found"

**Solution**:
- Windows: Try `py` instead of `python`
- Check Python is installed: `py --version` or `python3 --version`

---

## Video Processing Issues

### Issue: "Could not open video file: videos/cascade.mp4"

**Solutions**:
1. Check file exists: `dir videos\cascade.mp4` (Windows) or `ls videos/cascade.mp4`
2. Check file path is correct (should be `videos/cascade.mp4` relative to project root)
3. Try absolute path: Update script to use full path

### Issue: "No balls detected" or empty paths

**Causes**:
- Current detection uses HoughCircles (basic method)
- Video quality/lighting issues
- Ball color blends with background

**Solutions**:
1. This is expected - ball detection needs improvement (see Step 3.2 in guide)
2. Try adjusting HoughCircles parameters in `ball_tracker.py`
3. Consider switching to YOLO or color-based detection

---

## Import Errors

### Issue: "ModuleNotFoundError: No module named 'juggle_buddy'"

**Solution**:
1. Make sure you're in the project root directory
2. Install package in development mode: `pip install -e .`
3. Or add to PYTHONPATH: `set PYTHONPATH=%CD%` (Windows) or `export PYTHONPATH=$(pwd)` (Mac/Linux)

### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Solution**:
- Install OpenCV: `pip install opencv-python`
- Or reinstall all requirements: `pip install -r requirements.txt`

---

## Test Failures

### Issue: Tests fail with "AssertionError"

**For Step 1.1** (test_path_utils.py):
- These should pass - if they fail, check `juggle_buddy/path_utils.py` is correct

**For other tests**:
- Some tests may fail initially (expected)
- Follow the guide to implement missing functionality
- Tests are meant to guide development

---

## Performance Issues

### Issue: Video processing is very slow

**Solutions**:
1. Process fewer frames initially (test with first 100 frames)
2. Reduce video resolution before processing
3. Consider processing in batches

### Issue: Webcam not opening

**Solutions**:
1. Check camera permissions
2. Try different camera index: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`
3. Close other applications using the camera

---

## Getting Help

If you encounter issues not covered here:

1. Check the error message carefully
2. Verify you're on the correct step in `STEP_BY_STEP_GUIDE.md`
3. Check that all previous steps completed successfully
4. Review the code examples in the guide
