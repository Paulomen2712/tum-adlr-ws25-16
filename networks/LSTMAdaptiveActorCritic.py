import torch
from torch import nn
import torch.optim as optim
from networks.mlp import MLP, LSTM
import copy
from collections import deque
from networks.policy import AdaptivePolicy
from torch.distributions import MultivariateNormal

class LSTMAdaptiveActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_size=1, encoder_hidden_dims=[64, 32], lr=1e-5, std=0.5, history_len=10):
        super(LSTMAdaptiveActorCritic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_size = latent_size
        self.history_len = history_len
        self.std = std

        # History buffers
        self.obs_history = deque(maxlen=history_len)
        self.action_history = deque(maxlen=history_len)

        # LSTM Encoder
        self.lstm_input_dim = obs_dim + action_dim  # Concatenated observation and action
        self.encoder = LSTM(
            input_dim=self.lstm_input_dim,
            output_dim=latent_size,
            hidden_dims=encoder_hidden_dims,
            num_layers=len(encoder_hidden_dims),
            batch_first=True
        )

        # Optimizer
        self.optim = optim.Adam(self.parameters(), lr=lr)

        # Placeholder for actor (to be defined elsewhere)
        self.actor = None

        # Covariance matrix for action sampling
        self.cov_mat = torch.diag(torch.full((action_dim,), std))

    def clear_history(self):
        self.obs_history.clear()
        self.action_history.clear()

    def set_policy(self, policy):
        self.actor = copy.deepcopy(policy.actor)
        for param in self.actor.parameters():
            param.requires_grad = False

    def encode(self, obs):
        """
        Encode the observation and action history using the LSTM encoder.
        Apply masking to the observation before appending it to the history.
        
        Args:
            obs: Current observation tensor of shape (batch_size, obs_dim).
        
        Returns:
            Concatenated tensor of the masked observation and the encoded history.
        """
        # Ensure the observation has the same shape as the last one in the history
        if len(self.obs_history) > 0:
            expected_size = self.obs_history[-1].shape[0]
            current_size = obs.shape[0]

            if current_size < expected_size:
                padding = torch.zeros((expected_size - current_size, obs.shape[1]), device=obs.device)
                obs = torch.cat([obs, padding], dim=0)
            elif current_size > expected_size:
                obs = obs[:expected_size]

        # Apply masking to the observation
        masked_obs = obs.clone()
        masked_obs = masked_obs * torch.cat(
            [torch.ones_like(masked_obs[..., :-1]), torch.zeros_like(masked_obs[..., -1:])], dim=-1
        )

        # Append the masked observation to the history
        self.obs_history.append(masked_obs.clone())

        # Handling action history
        if len(self.action_history) < self.history_len:
            if len(self.action_history) > 0:
                padded_actions = [self.action_history[0].clone()] * (self.history_len - len(self.action_history))
            else:
                # Hard-coded action_dim in padding
                padded_actions = [torch.zeros((obs.shape[0], self.action_dim), device=obs.device)] * self.history_len
            all_actions = padded_actions + list(self.action_history)
        else:
            all_actions = list(self.action_history)

        # Handling observation history
        if len(self.obs_history) < self.history_len:
            if len(self.obs_history) > 0:
                padded_obs = [torch.zeros_like(obs)] * (self.history_len - len(self.obs_history))
            else:
                padded_obs = [torch.zeros_like(obs)] * self.history_len
            all_obs = padded_obs + list(self.obs_history)
        else:
            all_obs = list(self.obs_history)

        # Convert history to tensors
        all_obs_tensor = torch.stack(all_obs, dim=1)  # Shape: (batch_size, history_len, obs_dim)
        all_actions_tensor = torch.stack(all_actions, dim=1)  # Shape: (batch_size, history_len, action_dim)

        # Concatenate observations and actions
        lstm_input = torch.cat([all_obs_tensor, all_actions_tensor], dim=-1)  # Shape: (batch_size, history_len, obs_dim + action_dim)

        # Pass through LSTM encoder
        z = self.encoder(lstm_input)  # Shape: (batch_size, latent_size)

        # Return the concatenated masked observation and encoded history
        return torch.cat([masked_obs, z], dim=-1)


    @torch.no_grad()
    def sample_action(self, obs):
        ext_obs = self.encode(obs)
        mean = self.actor(ext_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()

        self.action_history.append(action.clone())

        return action

    def evaluate(self, obs):
        """
            Estimates the values of each observation, and the log probs of
            each action given the batch observations and actions. 
        """
        return self.encode(obs)[:,-1]

    def store_savestate(self, checkpoint_path):
        """
            Stores the model into the given directory.
        """
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

    def restore_savestate(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['optimizer_state_dict'])