BATCH_SIZE = 16
S1 = 7
S2 = 12
B = 5
C = 0
T = B * (5 + C)
IMG_WIDTH = 996
IMG_HEIGHT = 700
GRID_WIDTH = IMG_WIDTH // S2
GRID_HEIGHT = IMG_HEIGHT // S1
# MAX_BOX_WIDTH = IMG_WIDTH / 2
# MAX_BOX_HEIGHT = IMG_HEIGHT / 2
CONFIDENCE_THRESHOLD = 0.518
OUT_DIR = 'out'
