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

    def __init__(self, input_dim, output_dim, hidden_dims=[64], activation=nn.ELU, last_activation = None):
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
        if last_activation != None:
            layers.append(last_activation())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class DummyEncoder(nn.Module):
    def __init__(self):
        super(DummyEncoder, self).__init__()
    
    def forward(self, x):
        return x[:, -1]

class LSTM(nn.Module):
    """ LSTM-based sequence encoder. """

    def __init__(self, input_dim, output_dim, hidden_dims=[8], num_layers=1, batch_first=True):
        """
        Initialize parameters and build model.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output (latent space).
            hidden_dims (list): List of hidden layer dimensions for the LSTM.
            num_layers (int): Number of LSTM layers.
            batch_first (bool): If True, input tensors are expected to be (batch_size, seq_len, input_dim).
        """
        super(LSTM, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],  # Use the first hidden dimension for LSTM
            num_layers=num_layers,
            batch_first=batch_first
        )

        # Fully connected layer to map LSTM output to the desired output dimension
        self.fc = nn.Linear(hidden_dims[0], output_dim)

    def forward(self, x, hidden):
        """
        Forward pass through the LSTM.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)  # lstm_out shape: (batch_size, seq_len, hidden_size)

        # Map to output dimension
        output = self.fc(lstm_out)  # Shape: (batch_size, output_dim)

        return output, h_n, c_n