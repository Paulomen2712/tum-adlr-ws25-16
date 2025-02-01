import torch
from torch import nn
import torch.optim as optim
from networks.mlp import MLP
import copy
from torch.distributions import MultivariateNormal
class AdaptivePolicy(nn.Module):
    """ Actor Critic Model."""

    def __init__(self, obs_dim, action_dim, latent_size=1, encoder_hidden_dims=[64, 32], encoder_class = MLP, activation=nn.ELU, last_activation=None):
        """
            Initialize parameters and build model.
        """
        
        super(AdaptivePolicy, self).__init__()

        self.encoder = encoder_class(input_dim=obs_dim, output_dim=latent_size, hidden_dims=encoder_hidden_dims, activation=activation,last_activation=last_activation)


    def encode(self, obs):
        pass

    @torch.no_grad()
    def sample_action(self, obs):
        ext_obs = self.encode(obs)
        mean = self.actor(ext_obs)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        return dist.sample()
    
    def store_savestate(self, checkpoint_path):
        """
            Stores the model into the given directory.
        """
        pass

    def restore_savestate(self, checkpoint_path):
        pass