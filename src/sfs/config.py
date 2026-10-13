import torch
import torch.nn as nn

SEED = 1
N_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LOSS_FN = nn.CrossEntropyLoss
OPTIMIZER = torch.optim.Adam
