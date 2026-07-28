"""
Global configuration for Adaptive Learning Analytics.

This module contains all tunable parameters used throughout the
feature extraction, scoring and logging pipeline.
"""

from pathlib import Path

# ==============================================================================
# PROJECT PATHS
# ==============================================================================

ROOT_DIR = Path(__file__).resolve().parent

MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

MODEL_PATH = MODEL_DIR / "face_landmarker.task"

CSV_LOG_FILE = LOG_DIR / "metrics.csv"

LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ==============================================================================
# CAMERA
# ==============================================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

TARGET_FPS = 30

# ==============================================================================
# MEDIAPIPE
# ==============================================================================

MAX_NUM_FACES = 1

OUTPUT_BLENDSHAPES = True

OUTPUT_TRANSFORMATION_MATRIX = True

# ==============================================================================
# ROLLING WINDOWS
# ==============================================================================

FRAME_WINDOW_SECONDS = 60

SUMMARY_INTERVAL_SECONDS = 60

# ==============================================================================
# BLINK DETECTION
# ==============================================================================

EAR_THRESHOLD = 0.22

MIN_BLINK_FRAMES = 2

MAX_BLINK_FRAMES = 12

# ==============================================================================
# YAWN DETECTION
# ==============================================================================

MAR_THRESHOLD = 0.62

MIN_YAWN_FRAMES = 8

# ==============================================================================
# MOVEMENT
# ==============================================================================

FACE_MOVEMENT_THRESHOLD = 5.0

HEAD_ROTATION_THRESHOLD = 10.0

# ==============================================================================
# GAZE
# ==============================================================================

SCREEN_LOOK_YAW_LIMIT = 20.0

SCREEN_LOOK_PITCH_LIMIT = 15.0

# ==============================================================================
# TRACKING
# ==============================================================================

MIN_FACE_CONFIDENCE = 0.50

# ==============================================================================
# CSV HEADER
# ==============================================================================

CSV_COLUMNS = [

    "timestamp",

    "frames",

    "face_visible_pct",

    "screen_attention_pct",

    "mean_ear",
    "std_ear",

    "mean_mar",
    "std_mar",

    "blink_count",
    "blink_rate",

    "avg_blink_duration",

    "yawn_count",

    "pitch_mean",
    "pitch_std",

    "yaw_mean",
    "yaw_std",

    "roll_mean",
    "roll_std",

    "face_velocity",

    "movement_variance",

    "gaze_stability",

    "expression_energy",

    "attention_score",

    "fatigue_score",

    "focus_score",

    "interaction_score",

    "reading_score",

    "tracking_quality"
]