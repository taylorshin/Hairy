BATCH_SIZE = 16
# S1 = 7
S1 = 7
# S2 = 10
S2 = 12
T_temp = 32
B = 5
C = 0
T = B * (5 + C)
IMG_WIDTH = 996
IMG_HEIGHT = 700
# GRID_WIDTH = 100
GRID_WIDTH = IMG_WIDTH // S2
GRID_HEIGHT = IMG_HEIGHT // S1
MAX_BOX_WIDTH = 100
MAX_BOX_HEIGHT = 100 #500
CONFIDENCE_THRESHOLD = 0.58
OUT_DIR = 'out'
