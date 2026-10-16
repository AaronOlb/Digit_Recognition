import torch
import torch.nn as nn

SEED = 1
N_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LOSS_FN = nn.CrossEntropyLoss
OPTIMIZER = torch.optim.Adam


class CNNConfig:
    def __init__(
        self,
        channels=(1, 16, 32),
        kernel_size=3,
        padding=1,
        pooling_kernel_size=2,
        activation=nn.ReLU,
        pooling=nn.MaxPool2d,
    ):
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.pooling_kernel_size = pooling_kernel_size
        self.activation = activation
        self.pooling = pooling
