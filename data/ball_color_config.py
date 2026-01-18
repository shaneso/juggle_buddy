# Ball color HSV ranges (calibrated)
# Automatically loaded by juggle_buddy/ball_tracker.py

import numpy as np

BALL_COLOR_RANGES = {
    'red': (np.array([133, 113, 70]), np.array([179, 217, 229])),
    'blue': (np.array([100, 150, 0]), np.array([124, 255, 255])),
    'green': (np.array([5, 0, 0]), np.array([128, 177, 255])),
}
