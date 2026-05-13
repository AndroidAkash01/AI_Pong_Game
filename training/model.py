import torch
import torch.nn as nn


class PongMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[30]):
        super().__init__()

        layers = []
        prev_size = input_size

        # create hidden layers dynamically
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h

        # output layer (3 actions)
        layers.append(nn.Linear(prev_size, 3))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)