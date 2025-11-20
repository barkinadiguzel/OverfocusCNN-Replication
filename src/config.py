# Model params
NUM_CLASSES = 10       # output classes
INPUT_CHANNELS = 3     # RGB input
BLOCKS_PER_STAGE = [2,2,2]
FEATURE_MAPS = [64,128,256]

# SGDrop params
SGDROP_RHO = 0.1       # top % features to drop

# Training params (optional)
LR = 0.001
BATCH_SIZE = 32

