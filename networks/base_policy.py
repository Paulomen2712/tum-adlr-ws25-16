import torch
from torch import nn
import torch.optim as optim
from networks.mlp import MLP, Encoder
import numpy as np
from torch.distributions import MultivariateNormal
from networks.policy import AdaptivePolicy
# class ActorCriticWithEncoder(nn.Module):
#     """ Actor Critic Model."""

#     def __init__(self, obs_dim, action_dim, latent_size=1, hidden_dims=[64], encoder_hidden_dims=[64, 32], actor_lr=1e-5, critic_lr=1e-4, gamma=0.99, std=0.5):
#         """
#             Initialize parameters and build model.
#         """
        
#         super(ActorCriticWithEncoder, self).__init__()

#         self.encoder = MLP(obs_dim, latent_size, encoder_hidden_dims)

#         # Actor network (outputs probabilities for possible actions)
#         self.actor = MLP(obs_dim+latent_size, action_dim, hidden_dims)
        
#         # Critic network (outputs value estimate)
#         self.critic = MLP(obs_dim+latent_size, 1, hidden_dims)

#         self.register_buffer('cov_var', torch.full(size=(action_dim,), fill_value=std))
#         self.register_buffer('cov_mat', torch.diag(self.cov_var))

#         self.actor_optim = optim.Adam([*self.actor.parameters(), *self.encoder.parameters()], lr=actor_lr)
#         self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

#         scheduler_lambda = lambda epoch: gamma ** epoch
#         self.actor_scheduler= optim.lr_scheduler.LambdaLR(self.actor_optim, lr_lambda=scheduler_lambda)
#         self.critic_scheduler= optim.lr_scheduler.LambdaLR(self.critic_optim, lr_lambda=scheduler_lambda)

#     def encode(self, obs):
#         masked_obs = obs.clone()
#         masked_obs = masked_obs * torch.cat([torch.ones_like(masked_obs[..., :-1]), torch.zeros_like(masked_obs[..., -1:])], dim=-1)
#         z = self.encoder(obs)

#         return torch.cat([masked_obs, z], dim=-1)

#     @torch.no_grad()
#     def sample_action(self, obs):
#         ext_obs = self.encode(obs)
#         mean = self.actor(ext_obs)
#         dist = MultivariateNormal(mean, self.cov_mat)
#         return dist.sample()

#     @torch.no_grad()
#     def act(self, obs):
#         """
#             Samples an action from the actor/critic network.
#         """
#         ext_obs = self.encode(obs)
#         mean, values = self.actor(ext_obs), self.critic(ext_obs)
#         dist = MultivariateNormal(mean, self.cov_mat)
#         action = dist.sample()
#         log_prob = dist.log_prob(action)

#         return action, log_prob, values.view(-1)
    
#     @torch.no_grad()
#     def get_value(self, obs):
#         return self.critic(self.encode(obs)).squeeze()

#     def evaluate(self, obs, acts):
#         """
#             Estimates the values of each observation, and the log probs of
#             each action given the batch observations and actions. 
#         """
#         ext_obs = self.encode(obs)
#         mean, values = self.actor(ext_obs), self.critic(ext_obs.detach())
#         dist = MultivariateNormal(mean, self.cov_mat)
#         log_probs = dist.log_prob(acts)

#         return values.squeeze(), log_probs, dist.entropy()
    
#     def store_savestate(self, checkpoint_path):
#         """
#             Stores the model into the given directory.
#         """
#         checkpoint = {
#             'encoder_state_dict': self.encoder.state_dict(),
#             'actor_state_dict': self.actor.state_dict(),
#             'critic_state_dict': self.critic.state_dict(),
#             'actor_optimizer_state_dict': self.actor_optim.state_dict(),
#             'critic_optimizer_state_dict': self.critic_optim.state_dict(),
#             'actor_scheduler_state_dict': self.actor_scheduler.state_dict(),
#             'critic_scheduler_state_dict': self.critic_scheduler.state_dict()
#         }
#         torch.save(checkpoint, checkpoint_path)
#         print(f"Checkpoint saved at {checkpoint_path}")

#     def restore_savestate(self, checkpoint_path):
#         checkpoint = torch.load(checkpoint_path)
#         self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
#         self.actor.load_state_dict(checkpoint['actor_state_dict'])
#         self.critic.load_state_dict(checkpoint['critic_state_dict'])
#         self.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])
#         self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
#         self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
#         self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
#         print(f"Checkpoint restored from {checkpoint_path}")
class ActorCriticWithEncoder(AdaptivePolicy):
    """ Actor Critic Model."""

    def __init__(self, obs_dim, action_dim, latent_size=1, hidden_dims=[64], encoder_hidden_dims=[64, 32], actor_lr=1e-5, critic_lr=1e-4, std=0.5):
        """
            Initialize parameters and build model.
        """
        
        super(ActorCriticWithEncoder, self).__init__(obs_dim, action_dim, latent_size, encoder_hidden_dims, std)

        # Actor network (outputs probabilities for possible actions)
        self.actor = MLP(obs_dim+latent_size, action_dim, hidden_dims)
        
        # Critic network (outputs value estimate)
        self.critic = MLP(obs_dim+latent_size, 1, hidden_dims)


        self.actor_optim = optim.Adam([*self.actor.parameters(), *self.encoder.parameters()], lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def encode(self, obs, apply_masking = False):
        obs_clone = obs.clone()
        if(apply_masking):
            obs_clone = obs_clone * torch.cat([torch.ones_like(obs_clone[..., :-1]), torch.zeros_like(obs_clone[..., -1:])], dim=-1)

        z = self.encoder(obs)

        return torch.cat([obs_clone, z], dim=-1)

    @torch.no_grad()
    def act(self, obs):
        """
            Samples an action from the actor/critic network.
        """
        ext_obs = self.encode(obs)
        mean, values = self.actor(ext_obs), self.critic(ext_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, values.view(-1)
    
    @torch.no_grad()
    def get_value(self, obs):
        return self.critic(self.encode(obs)).squeeze()

    def evaluate(self, obs, acts):
        """
            Estimates the values of each observation, and the log probs of
            each action given the batch observations and actions. 
        """
        true_z = obs[:, -1].unsqueeze(1).flatten()

        ext_obs = self.encode(obs)

        encoded_z = ext_obs[:, -1].unsqueeze(1).flatten()

        mean, values = self.actor(ext_obs), self.critic(ext_obs.detach())
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(acts)

        return values.squeeze(), log_probs, dist.entropy(), true_z, encoded_z
    
    def store_savestate(self, checkpoint_path):
        """
            Stores the model into the given directory.
        """
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optim.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict(),
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def restore_savestate(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
        self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
        print(f"Checkpoint restored from {checkpoint_path}")