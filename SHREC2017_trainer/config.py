import os
import logging

# Setup logging
_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Dataset paths and configurations
MAIN_FOLDER = "./"  # Current directory as main directory

# 14 gesture classes
GESTURE_NAME = [
    "Grab", "Tap", "Expand", "Pinch", "Rotation CW", "Rotation CCW",
    "Swipe Right", "Swipe Left", "Swipe Up", "Swipe Down", "Swipe X",
    "Swipe V", "Swipe +", "Shake"
]

# Full 28 classes (14 gestures × 2 finger configurations)
FULL_GESTURE_NAME = []
for gesture in GESTURE_NAME:
    FULL_GESTURE_NAME.append(f"{gesture} (one finger)")
    FULL_GESTURE_NAME.append(f"{gesture} (whole hand)")

# Define joint names
# SHREC2017 skeleton structure includes 22 joints, each with 3D coordinates
JOINT_NAME = [
    "palm",
    "thumb_base", "thumb_middle", "thumb_end",
    "index_base", "index_middle", "index_middle2", "index_end",
    "middle_base", "middle_middle", "middle_middle2", "middle_end",
    "ring_base", "ring_middle", "ring_middle2", "ring_end",
    "pinky_base", "pinky_middle", "pinky_middle2", "pinky_end",
    "wrist1", "wrist2"
]

# Default using full 28 classes, can be modified via command line arguments
SELECTED_GESTURE_NAME = FULL_GESTURE_NAME
SELECTED_JOINT_NAME = JOINT_NAME

# Data dimensions
JOINTS_PER_FRAME = 22  # Number of joints per frame
COORDS_PER_JOINT = 3   # Number of coordinates per joint (x,y,z)
INPUT_DIM = len(SELECTED_JOINT_NAME) * COORDS_PER_JOINT  # Input dimension=22*3=66
OUTPUT_DIM = len(SELECTED_GESTURE_NAME)  # Output dimension=28 (number of classes)
MAX_LENGTH = 100  # Maximum sequence length
BATCH_SIZE = 8   # Batch size

# Training settings
MODEL_SAVE_DIR = "./models" 