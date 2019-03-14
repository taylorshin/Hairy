import os

# Files and directories
OUT_DIR = 'out'
DATA_DIR = 'data'
MODEL_DIR = os.path.join(OUT_DIR, 'model.h5')
LOG_DIR = os.path.join(OUT_DIR, 'logs')
PREDICT_DIR = os.path.join(OUT_DIR, 'predictions/experiment-data_aug/mse_aug_test_J/')
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
PIXEL_SHIFT_X = 10
PIXEL_SHIFT_Y = 20
DATA_AUG_PROB = 0.8
DOWNSCALE_FACTOR = 2

# Miscellaneous
IMG_WIDTH = 1000 // DOWNSCALE_FACTOR
IMG_HEIGHT = 700 // DOWNSCALE_FACTOR
IMG_CHANNELS = 11
GRID_WIDTH = IMG_WIDTH // S2
GRID_HEIGHT = IMG_HEIGHT // S1
CONFIDENCE_THRESHOLD = 0.5
LABEL_FRAME_INTERVAL = 20
CONTEXT_FRAMES = 5

# TODO: Figure out way to remove need for this constant
NUM_ITEMS_PER_DIR = 50
