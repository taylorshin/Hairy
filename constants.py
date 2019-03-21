import os

# Files and directories
OUT_DIR = 'out'
DATA_DIR = 'data'
MODEL_DIR = os.path.join(OUT_DIR, 'model.h5')
LOG_DIR = os.path.join(OUT_DIR, 'logs')
PREDICT_DIR = os.path.join(OUT_DIR, 'predictions')
PLOT_FILE = os.path.join(OUT_DIR, 'loss.png')

# Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# Constants for YOLO algorithm
S1 = 8
S2 = 13
B = 1
C = 0
T = B * (5 + C)
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5

# Data augmentation
DOWNSCALE_FACTOR = 1
PIXEL_SHIFT_X = 50 // DOWNSCALE_FACTOR
PIXEL_SHIFT_Y = 50 // DOWNSCALE_FACTOR
DATA_AUG_PROB = 0.8

# Miscellaneous
IMG_WIDTH = 1000 // DOWNSCALE_FACTOR # 500
IMG_HEIGHT = 700 // DOWNSCALE_FACTOR # 350
CONTEXT_FRAMES = 3
IMG_DEPTH = CONTEXT_FRAMES + 1 + CONTEXT_FRAMES
GRID_WIDTH = IMG_WIDTH // S2
GRID_HEIGHT = IMG_HEIGHT // S1
CONFIDENCE_THRESHOLD = 0.5
LABEL_FRAME_INTERVAL = 20

# KERNEL_DEPTH = IMG_DEPTH

# TODO: Figure out way to remove need for this constant
NUM_ITEMS_PER_DIR = 50
