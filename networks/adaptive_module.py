import torch
from torch import nn
import torch.optim as optim
from networks.mlp import MLP
import copy
from collections import deque
from networks.policy import AdaptivePolicy
from torch.distributions import MultivariateNormal
class AdaptiveActorCritic(AdaptivePolicy):
    """ Actor Critic Model."""

    def __init__(self, obs_dim, action_dim, latent_size=1, encoder_hidden_dims=[64, 32], lr=1e-5, history_len=10, activation=nn.ELU):
        """
            Initialize parameters and build model.
        """
        
        super(AdaptiveActorCritic, self).__init__(obs_dim, action_dim, latent_size, encoder_hidden_dims)
        self.action_dim = action_dim 
        self.history_len = history_len
        self.obs_history = deque(maxlen=history_len)
        self.action_history = deque(maxlen=history_len)

        self.encoder_input_dim = (obs_dim  * history_len) + (action_dim * history_len)
        self.encoder = MLP(self.encoder_input_dim, latent_size, encoder_hidden_dims, activation=activation)

        self.optim = optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor = None
        self.actor_logstd = None

    def clear_history(self):
        self.obs_history.clear()
        self.action_history.clear()

    def set_policy(self, policy):
        self.actor = copy.deepcopy(policy.actor)
        for param in self.actor.parameters():
            param.requires_grad = False
        self.actor_logstd = copy.deepcopy(policy.actor_logstd)
        self.actor_logstd.requires_grad = False

    def encode(self, obs):
        # Append the masked observation to the history
        mask = torch.ones_like(obs)  
        mask[..., -1] = 0  
        obs_clone = obs.clone() * mask
        # obs_clone = obs.clone()
        if not self.obs_history:
            self.obs_history.extend([torch.zeros_like(obs)] * self.history_len)
            self.obs_history.append(obs_clone)
        if not self.action_history:
            self.action_history.extend([torch.zeros((*obs.shape[:-1], self.action_dim), device=obs.device)] * self.history_len)

        history_tensor = torch.cat(list(self.obs_history) + list(self.action_history), dim=1)

        # # Encode the history tensor
        z = self.encoder(history_tensor)

        # Return the concatenated masked observation and encoded history
        return torch.cat([obs_clone, z], dim=-1)


    @torch.no_grad
    def sample_action(self, obs):
        ext_obs = self.encode(obs)
        mean = self.actor(ext_obs)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        action = dist.sample()

        self.action_history.append(action.clone())

        return action, ext_obs[..., -1]

    def evaluate(self, obs, action):
        """
            Estimates the values of each observation, and the log probs of
            each action given the batch observations and actions. 
        """
        z = self.encode(obs)[:,-1]
        self.action_history.append(action)
        return z