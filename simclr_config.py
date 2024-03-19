# Training Hyperparameters
BATCH_SIZE = 16
INPUT_SIZE = 512
MAX_EPOCHS = 10
IMAGE_MEAN = [0.4401, 0.5126, 0.3948] 
IMAGE_STD = [0.1959, 0.2017, 0.1640]
Z_DIMS = 128 # Embedding size for simCLR hidden state (zi)
LR = 6e-2 # Learning rate for optimizer