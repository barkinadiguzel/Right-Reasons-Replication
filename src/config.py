import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_DIM = 16
HIDDEN_DIM = 64
NUM_CLASSES = 3

LAMBDA_1 = 10.0   # Right reason weight
LAMBDA_2 = 1e-4   # Regularization
