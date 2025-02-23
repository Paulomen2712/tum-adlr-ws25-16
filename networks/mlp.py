from torch import nn

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
    """
    Encoder that maps (x1, ..., xn) -> xn

    Used for training against real disturbance value without having to change the ppo algorithm
    """
    def __init__(self, **args):
        super(DummyEncoder, self).__init__()
    
    def forward(self, x):
        return x[..., -1].unsqueeze(-1)

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

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],  # Use the first hidden dimension for LSTM for compatibility with other encoder classes
            num_layers=num_layers,
            batch_first=batch_first
        )

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

        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        output = self.fc(lstm_out)  

        return output, h_n, c_n