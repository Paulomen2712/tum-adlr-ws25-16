import torch
from torch import nn
import torch.optim as optim
from networks.mlp import MLP
import copy
from torch.distributions import MultivariateNormal
class AdaptivePolicy(nn.Module):
    """ Actor Critic Model."""

    def __init__(self, obs_dim, action_dim, latent_size=1, encoder_hidden_dims=[64, 32], std = 0.5):
        """
            Initialize parameters and build model.
        """
        
        super(AdaptivePolicy, self).__init__()

        self.encoder = MLP(obs_dim, latent_size, encoder_hidden_dims)

        self.register_buffer('cov_var', torch.full(size=(action_dim,), fill_value=std))
        self.register_buffer('cov_mat', torch.diag(self.cov_var))


    def encode(self, obs):
        pass

    @torch.no_grad()
    def sample_action(self, obs):
        ext_obs = self.encode(obs)
        mean = self.actor(ext_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        return dist.sample()
    
    def store_savestate(self, checkpoint_path):
        """
            Stores the model into the given directory.
        """
        pass

    def restore_savestate(self, checkpoint_path):
        pass