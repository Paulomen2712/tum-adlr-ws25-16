import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """ Actor State Encoder. """

    def __init__(self, state_size, latent_size, fc1_units=128, fc2_units=64):
        """
            Initialize parameters and build model.

        """
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, latent_size)

    def forward(self, state):
        """
            Build a network that maps state -> latent representation.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLP(nn.Module):
    """ Fully connected feedforward network with a hidden layer. """

    def __init__(self, input_dim, output_dim, hidden_dims=[64], activation=nn.ReLU):
        """
            Initialize parameters and build model.
        """
        super(MLP, self).__init__()
		
        layer_dims = [input_dim] + hidden_dims + [output_dim]

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(activation())   
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)