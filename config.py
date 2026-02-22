"""
Configuration - Two-Lane Bidirectional Autonomous Driving Simulation
"""
import pygame

# ==================== DISPLAY ====================
pygame.init()
WIDTH  = 1280
HEIGHT = 720
FPS    = 60

# ==================== ROAD LAYOUT ====================
ROAD_WIDTH  = 420
LANE_COUNT  = 2
LANE_WIDTH  = ROAD_WIDTH // LANE_COUNT   # 210 px
ROAD_LEFT   = (WIDTH - ROAD_WIDTH) // 2
ROAD_RIGHT  = ROAD_LEFT + ROAD_WIDTH

LANE_CENTERS = [
    ROAD_LEFT + LANE_WIDTH // 2,
    ROAD_LEFT + LANE_WIDTH + LANE_WIDTH // 2,
]

# ==================== VEHICLE SIZES ====================
VEHICLE_SCALE = 0.38
CAR_WIDTH     = int(LANE_WIDTH * VEHICLE_SCALE)   # ~80 px
_CAR_H_REF    = int(CAR_WIDTH * 1.75)             # ~140 px

# ==================== VEHICLE DYNAMICS ====================
CRUISE_SPEED       = 3.0
MAX_SPEED          = 3.8
OVERTAKE_SPEED     = 3.5
MIN_FOLLOW_SPEED   = 0.6

ROTATION_SPEED     = 1.8
MAX_ROTATION_ANGLE = 25
ACCELERATION       = 0.04
DECELERATION       = 0.06
BRAKE_FORCE        = 0.12
FOLLOW_SPEED_MARGIN = 0.3

# ==================== TEMPORAL ====================
SEQUENCE_LENGTH    = 10
DECISION_FREQUENCY = 6
TIME_PER_FRAME     = 1.0 / FPS

# ==================== TRAFFIC ====================
SAME_DIR_NUM_CARS   = 4
SAME_DIR_SPEED_MIN  = 1.3
SAME_DIR_SPEED_MAX  = 2.0

SAME_DIR_MIN_GAP    = int(_CAR_H_REF * 2.5)   # Realistic gap to avoid continuous blocking
SAME_DIR_SPAWN_GAP  = int(_CAR_H_REF * 4.5)   # Larger spawn gap for traffic gaps
SAME_DIR_SPAWN_PROB = 0.012                    # Slightly lower for clearer windows

# Oncoming: only 1 active car at a time with HUGE enforced gaps
# so there are always long clear windows to overtake through.
ONCOMING_NUM_CARS   = 1
ONCOMING_SPEED_MIN  = 2.0
ONCOMING_SPEED_MAX  = 2.8

ONCOMING_MIN_GAP    = int(_CAR_H_REF * 12.0)  # ~1680px gap between cars
ONCOMING_SPAWN_GAP  = int(_CAR_H_REF * 15.0)
ONCOMING_SPAWN_PROB = 0.006                    # slow spawn = long quiet stretches

# ==================== SAFETY / OVERTAKE THRESHOLDS ====================
SAFETY_DISTANCE_THRESHOLD       = int(_CAR_H_REF * 2.0)
TTC_CRITICAL                    = 1.5
TTC_WARNING                     = 3.0
COLLISION_PROBABILITY_THRESHOLD = 0.3

OVERTAKE_TRIGGER_GAP  = int(_CAR_H_REF * 2.2)
OVERTAKE_COMMIT_GAP   = int(_CAR_H_REF * 1.2)

# Clearance zone required before moving into oncoming lane
ONCOMING_CLEAR_AHEAD  = int(_CAR_H_REF * 6.0)
ONCOMING_CLEAR_BEHIND = int(_CAR_H_REF * 2.5)

# Once lane change starts, commit for at least this many frames (no oscillate)
OVERTAKE_COMMIT_MIN_FRAMES = 45

# After overtake: no same-dir spawn in front of ego for this long (frames)
POST_OVERTAKE_SPAWN_COOLDOWN_FRAMES = 180
# Minimum distance ahead of ego for new same-dir spawn (px)
MIN_SPAWN_DIST_AHEAD = int(_CAR_H_REF * 5.0)
# During post-overtake cooldown, spawns must be at least this far ahead (px)
POST_OVERTAKE_SPAWN_BUFFER = int(_CAR_H_REF * 8.0)

EMERGENCY_BRAKE_DIST  = int(_CAR_H_REF * 0.5)

# ==================== SENSOR CONFIGURATION ====================
SENSOR_CONFIG = {
    'front':     {'count': 5, 'length': 700, 'spread': 50},
    'diagonal':  {'count': 3, 'length': 500, 'spread': 22, 'offset': 55},
    'rear':      {'count': 3, 'length': 350, 'spread': 28},
    'rear_diag': {'count': 2, 'length': 350, 'angle': 45},
}

# ==================== MODEL / TRAINING ====================
LSTM_UNITS_LAYER1    = 128
LSTM_UNITS_LAYER2    = 64
DROPOUT_RATE         = 0.3
DENSE_UNITS          = 32
BATCH_SIZE           = 32
EPOCHS               = 30
VALIDATION_SPLIT     = 0.2
MIN_ACCURACY_THRESHOLD = 0.95

TRAINING_DATA_FILE   = "training_data.csv"
MODEL_FILE           = "autonomous_brain_lstm.h5"
LABEL_ENCODER_FILE   = "label_encoder.pkl"
SCALER_FILE          = "feature_scaler.pkl"

# ==================== COLORS ====================
WHITE      = (255, 255, 255)
BLACK      = (  0,   0,   0)
YELLOW     = (255, 215,   0)
GRAY       = (169, 169, 169)
DARK_GRAY  = ( 50,  50,  50)
RED        = (220,  30,  30)
BLUE       = ( 30, 100, 220)
CYAN       = (  0, 220, 220)
MAGENTA    = (220,   0, 220)
GREEN      = ( 50, 200,  50)
ORANGE     = (255, 140,   0)
RED_LASER  = (255,  60,  60)
BLUE_LASER = (  0, 230, 230)
GRASS_COL  = ( 34,  90,  34)
ASPHALT    = ( 48,  48,  52)
