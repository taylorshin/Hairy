import os

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
OUT_DIR = 'out'
DATA_DIR = 'data'
MODEL_DIR = os.path.join(OUT_DIR, 'model.h5')
LOG_DIR = os.path.join(OUT_DIR, 'logs')
PREDICT_DIR = os.path.join(OUT_DIR, 'predictions/train_set_G_dataaug_mse_150/')

# Constants for YOLO algorithm
S1 = 6
S2 = 11
B = 1
C = 0
T = B * (5 + C)
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5

IMG_WIDTH = 1000
IMG_HEIGHT = 700
IMG_CHANNELS = 11
GRID_WIDTH = IMG_WIDTH // S2
GRID_HEIGHT = IMG_HEIGHT // S1
CONFIDENCE_THRESHOLD = 0.5
LABEL_FRAME_INTERVAL = 20
CONTEXT_FRAMES = 5
PIXEL_SHIFT_X = 100
PIXEL_SHIFT_Y = 200
AUG_PROB = 0.2

# TODO: Figure out way to remove need for this constant
NUM_ITEMS_PER_DIR = 50
