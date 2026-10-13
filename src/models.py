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


def make_CNN():
    pass

