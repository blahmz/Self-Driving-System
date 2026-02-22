"""
Configuration file for Self-Driving Car Simulation
Contains all constants and hyperparameters
"""

import pygame

# ==================== DISPLAY SETTINGS ====================
pygame.init()
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
FPS = 40

# ==================== ROAD SETTINGS ====================
ROAD_WIDTH_RATIO = 0.5  # 50% of screen height
ROAD_WIDTH = int(HEIGHT * ROAD_WIDTH_RATIO)
LANE_COUNT = 3
LANE_WIDTH = ROAD_WIDTH // LANE_COUNT
CAR_WIDTH = int(0.65 * LANE_WIDTH)

# ==================== VEHICLE DYNAMICS ====================
BASE_CAR_SPEED = 4
ROTATION_SPEED = 2
MAX_ROTATION_ANGLE = 35
ACCELERATION = 0.1
DECELERATION = 0.075
MAX_SPEED = 6

# ==================== TEMPORAL SETTINGS ====================
SEQUENCE_LENGTH = 10  # Number of timesteps to look back
DECISION_FREQUENCY = 5  # Make decision every N frames
TIME_PER_FRAME = 1.0 / FPS  # seconds per frame

# ==================== TRAFFIC DENSITY SETTINGS ====================
DENSITY_LEVELS = {
    'LOW': {'num_cars': 3, 'spawn_prob': 0.01, 'min_gap': 400},
    'MEDIUM': {'num_cars': 6, 'spawn_prob': 0.03, 'min_gap': 250},
    'HIGH': {'num_cars': 10, 'spawn_prob': 0.05, 'min_gap': 150}
}
DEFAULT_DENSITY = 'MEDIUM'

# ==================== SAFETY CONSTRAINTS ====================
SAFETY_DISTANCE_THRESHOLD = 200  # pixels
TTC_CRITICAL = 2.0  # seconds
TTC_WARNING = 4.0  # seconds
COLLISION_PROBABILITY_THRESHOLD = 0.3

# ==================== SENSOR CONFIGURATION ====================
SENSOR_CONFIG = {
    'front': {'count': 5, 'length': 450, 'spread': 60},  # degrees
    'diagonal': {'count': 3, 'length': 300, 'spread': 22.5, 'offset': 60},
    'rear': {'count': 3, 'length': 250, 'spread': 30},
    'rear_diagonal': {'count': 2, 'length': 250, 'angle': 45}
}

# ==================== MODEL SETTINGS ====================
LSTM_UNITS_LAYER1 = 128
LSTM_UNITS_LAYER2 = 64
DROPOUT_RATE = 0.3
DENSE_UNITS = 32
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.2
MIN_ACCURACY_THRESHOLD = 0.99

# ==================== FSM STATES ====================
FSM_STATES = [
    'LANE_KEEPING',
    'LANE_CHANGE_LEFT',
    'LANE_CHANGE_RIGHT',
    'EMERGENCY_BRAKE',
    'ADAPTIVE_CRUISE'
]

# ==================== COLORS ====================
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
GRAY = (169, 169, 169)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)
RED_LASER = (255, 50, 50)
BLUE_LASER = (0, 255, 255)

# ==================== FILE PATHS ====================
TRAINING_DATA_FILE = "training_data.csv"
MODEL_FILE = "autonomous_brain_lstm.h5"
LABEL_ENCODER_FILE = "label_encoder.pkl"
SCALER_FILE = "feature_scaler.pkl"
DENSITY_LOG_FILE = "density_log.csv"

# ==================== EXTERNAL MODEL INTEGRATION ====================
# Use the model + scaler + labels from the separate `autonomous_car` project.
# This preserves this project's UI/HUD/FSM while swapping the AI "brain".
USE_EXTERNAL_AUTONOMOUS_CAR_MODEL = True

# Paths to external artifacts (from `E:\Project\autonomous_car\`)
EXTERNAL_MODEL_FILE = r"E:\Project\autonomous_car\final_lstm_model.h5"
EXTERNAL_SCALER_FILE = r"E:\Project\autonomous_car\scaler.pkl"
EXTERNAL_LABELS_FILE = r"E:\Project\autonomous_car\label_classes.npy"

# ==================== AUTO-RUN CONFIGURATION ====================
AUTO_RUN_ENABLED = True  # Set to False to disable auto-run
AUTO_RUN_TOTAL_TIME = 300  # Total run time in seconds (5 minutes)
AUTO_RUN_DENSITY_SCHEDULE = [
    {'duration': 60, 'density': 'LOW'},      # 0-60s: LOW
    {'duration': 120, 'density': 'MEDIUM'},  # 60-180s: MEDIUM
    {'duration': 120, 'density': 'HIGH'}     # 180-300s: HIGH
]