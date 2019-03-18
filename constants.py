import os

# Files and directories
OUT_DIR = 'out'
DATA_DIR = 'data'
MODEL_DIR = os.path.join(OUT_DIR, 'model.h5')
LOG_DIR = os.path.join(OUT_DIR, 'logs')
PREDICT_DIR = os.path.join(OUT_DIR, 'predictions')
PLOT_FILE = os.path.join(OUT_DIR, 'loss.png')

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# Constants for YOLO algorithm
S1 = 7
S2 = 11
B = 1
C = 0
T = B * (5 + C)
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5

# Data augmentation
DOWNSCALE_FACTOR = 2
PIXEL_SHIFT_X = 50 // DOWNSCALE_FACTOR
PIXEL_SHIFT_Y = 100 // DOWNSCALE_FACTOR
DATA_AUG_PROB = 0.5

# Miscellaneous
IMG_WIDTH = 1000 // DOWNSCALE_FACTOR # 500
IMG_HEIGHT = 700 // DOWNSCALE_FACTOR # 350
IMG_CHANNELS = 11
GRID_WIDTH = IMG_WIDTH // S2
GRID_HEIGHT = IMG_HEIGHT // S1
CONFIDENCE_THRESHOLD = 0.5
LABEL_FRAME_INTERVAL = 20
CONTEXT_FRAMES = 5

# TODO: Figure out way to remove need for this constant
NUM_ITEMS_PER_DIR = 50
