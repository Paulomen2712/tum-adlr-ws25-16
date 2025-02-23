import torch
from torch import nn
import torch.optim as optim
from networks.mlp import LSTM
import copy

class LSTMAdaptiveActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_size=1, encoder_hidden_dims=[64, 32], num_layers = 1, lr=1e-5, std=0.5, history_len=10, activation =None):
        super(LSTMAdaptiveActorCritic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_size = latent_size
        self.history_len = history_len
        self.std = std

        # LSTM Encoder
        self.lstm_input_dim = obs_dim + action_dim  # Concatenated observation and action
        self.encoder = LSTM(
            input_dim=self.lstm_input_dim,
            output_dim=latent_size,
            hidden_dims=encoder_hidden_dims,
            num_layers=num_layers,
            batch_first=True
        )
        self.hidden_shape = encoder_hidden_dims[0]
        self.h = torch.zeros((1,1,self.hidden_shape))
        self.c = torch.zeros((1,1,self.hidden_shape))
        self.prev_action = torch.zeros((1, self.action_dim))

        # Optimizer
        self.optim = optim.Adam(self.encoder.parameters(), lr=lr)

        # Placeholder for actor (to be defined elsewhere)
        self.actor = None
        self.actor_logstd = None

    def clear_history(self, indexes = None):
        """Resets hidden states."""
        if indexes is None:
            self.h = (torch.zeros_like(self.h))
            self.c = (torch.zeros_like(self.c))
            self.prev_action = None
            return
        self.h[indexes] = 0
        self.c[indexes]  = 0
        self.prev_action[indexes]  = 0

    def set_policy(self, policy):
        """Sets the policy to a frozen copy of a trained one"""
        self.actor = copy.deepcopy(policy.actor)
        for param in self.actor.parameters():
            param.requires_grad = False
        self.actor_logstd = copy.deepcopy(policy.actor_logstd)
        self.actor_logstd.requires_grad = False

    def encode(self, obs, action = None):
        """
        Encode the observation and action history using the LSTM encoder.
        Apply masking to the observation before appending it to the history.
        """
        if self.h.shape[:-1] != obs.shape[:-1]:
            #size missmatch means new setting -> reset hidden states
            self.h = torch.zeros((1, obs.shape[0], self.hidden_shape)).to(obs.device)
            self.c = torch.zeros((1, obs.shape[0], self.hidden_shape)).to(obs.device)
            self.prev_action = torch.zeros((*obs.shape[0:-1], self.action_dim)).to(obs.device)
        mask = torch.ones_like(obs)  
        mask[..., -1] = 0  
        obs_clone = obs.clone() * mask
        if action is None:
            action = self.prev_action = torch.zeros((*obs.shape[0:-1], self.action_dim)).to(obs.device)
        # Pass through LSTM encoder
        z, h_n, c_n = self.encoder(torch.cat([obs_clone, action], dim=-1), (self.h.to(obs.device), self.c.to(obs.device)))
        self.h = h_n
        self.c = c_n
        # Return the concatenated masked observation and encoded history
        return torch.cat([obs_clone, z], dim=-1)

    @torch.no_grad
    def sample_action(self, obs):
        """Samples an action. Also returns encoded extrinsics"""
        obs = obs.unsqueeze(1)
        ext_obs = self.encode(obs).squeeze(1)
        mean = self.actor(ext_obs)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        action = dist.sample()
        self.prev_action = action
        return action, ext_obs[..., -1]

    def evaluate(self, obs, action=None):
        """
            Estimates the values of each observation, and the log probs of
            each action given the batch observations and actions. 
        """
        return self.encode(obs, action)[...,-1]
    