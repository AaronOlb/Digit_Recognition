import torch.nn as nn

#Note that for now, we write make_MLP and make_CNN with the assumption that we only use the digit_recognizer dataset
#This will be generalised later on. 


def make_MLP(hidden_sizes = (784,), activation = nn.ReLU):
    
    in_dim = 784
    layers = []

    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, h))
        layers.append(activation())
        in_dim = h

    layers.append(nn.Linear(in_dim, 10))

    return nn.Sequential(*layers)


def make_CNN(config):
    layers = [nn.Unflatten(1, (1,28,28))]
    for i in range(len(config.channels)-1):
        layers.append(nn.Conv2d(config.channels[i], config.channels[i+1], kernel_size = config.kernel_size, padding = config.padding))
        layers.append(config.activation())
        layers.append(config.pooling(kernel_size = config.pooling_kernel_size))
    
    layers.extend([nn.Flatten(), nn.LazyLinear(10)])

    return nn.Sequential(*layers)

