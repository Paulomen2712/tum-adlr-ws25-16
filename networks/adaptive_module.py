import torch
from torch import nn
import torch.optim as optim
from networks.mlp import MLP
import copy
from networks.policy import AdaptivePolicy
class AdaptiveActorCritic(AdaptivePolicy):
    """ Actor Critic Model."""

    def __init__(self, obs_dim, action_dim, latent_size=1, encoder_hidden_dims=[64, 32], lr=1e-5, std = 0.5):
        """
            Initialize parameters and build model.
        """
        
        super(AdaptiveActorCritic, self).__init__(obs_dim, action_dim, latent_size, encoder_hidden_dims, std)
        self.optim = optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor = None

    def set_policy(self, policy):
        self.actor = copy.deepcopy(policy.actor)
        for param in self.actor.parameters():
            param.requires_grad = False

    def encode(self, obs):
        masked_obs = obs.clone()
        masked_obs = masked_obs * torch.cat([torch.ones_like(masked_obs[..., :-1]), torch.zeros_like(masked_obs[..., -1:])], dim=-1)
        z = self.encoder(obs)

        return torch.cat([masked_obs, z], dim=-1)

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