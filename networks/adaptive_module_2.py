import torch
from torch import nn
import torch.optim as optim
from networks.mlp import MLP
import copy
from collections import deque
from torch.distributions import MultivariateNormal

class AdaptiveActorCritic(nn.Module):
    """ Actor Critic Model."""

    def __init__(self, obs_dim, action_dim, latent_size=1, encoder_hidden_dims=[64, 32], lr=1e-5, std = 0.5, history_len=4):
        """
            Initialize parameters and build model.
        """
        
        super(AdaptiveActorCritic, self).__init__()

        self.history_len = history_len
        self.obs_history = deque(maxlen=history_len)
        self.action_history = deque(maxlen=history_len)

        adpt_module_input_dim = (obs_dim * history_len) + (action_dim * history_len)
        self.adpt_module = MLP(adpt_module_input_dim, latent_size, encoder_hidden_dims)

        self.register_buffer('cov_var', torch.full(size=(action_dim,), fill_value=std))
        self.register_buffer('cov_mat', torch.diag(self.cov_var))

        self.optim = optim.Adam(self.adpt_module.parameters(), lr=lr)
        self.base_encoder = None
        self.actor = None

    def set_policy(self, policy):
        self.actor = copy.deepcopy(policy.actor)
        self.base_encoder = copy.deepcopy(policy.encoder)
        for param in [*self.actor.parameters(), *self.base_encoder.parameters()]:
            param.requires_grad = False

    def encode(self, obs):
        masked_obs = obs.clone()
        masked_obs = masked_obs * torch.cat([torch.ones_like(masked_obs[..., :-1]), torch.zeros_like(masked_obs[..., -1:])], dim=-1)

        self.obs_history.append(masked_obs.clone())
        if len(self.action_history) < self.history_len:
            padded_actions = [torch.zeros_like(self.action_history[0])] * (self.history_len - len(self.action_history))
            all_actions = padded_actions + list(self.action_history)
        else:
            all_actions = list(self.action_history)

        if len(self.obs_history) < self.history_len:
            padded_obs = [torch.zeros_like(self.obs_history[0])] * (self.history_len - len(self.obs_history))
            all_obs = padded_obs + list(self.obs_history)
        else:
            all_obs = list(self.obs_history)

        history_tensor = torch.cat([torch.cat(all_obs, dim=-1), torch.cat(all_actions, dim=-1)], dim=-1)
        
        z = self.adpt_module(history_tensor)

        return torch.cat([masked_obs, z], dim=-1)

    @torch.no_grad()
    def sample_action(self, obs):
        ext_obs = self.encode(obs)
        mean = self.actor(ext_obs)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        action = dist.sample()

        self.action_history.append(action.clone())

        return action

    @torch.no_grad()
    def act(self, obs):
        """
            Samples an action from the actor/critic network.
        """
        ext_obs = self.encode(obs)
        mean = self.actor(ext_obs)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.action_history.append(action.clone())

        return action, log_prob

    def evaluate(self, obs, acts):
        """
            Estimates the values of each observation, and the log probs of
            each action given the batch observations and actions. 
        """
        ext_obs = self.encode(obs)
        mean, values = self.actor(ext_obs), self.critic(ext_obs.detach())
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        log_probs = dist.log_prob(acts)

        return values.squeeze()
    
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