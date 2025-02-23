import torch
from torch import nn
import torch.optim as optim
from networks.mlp import MLP

class ActorCriticWithEncoder(nn.Module):
    """ Actor Critic Model with encoded extrinsics concatenated to input."""

    def __init__(self, obs_dim, action_dim, latent_size=1, hidden_dims=[64], encoder_hidden_dims=[64, 32], encoder_class = MLP, lr=1e-5, activation=nn.ELU,last_activation=nn.Tanh):
        """
            Initialize parameters and build model.
        """
        super(ActorCriticWithEncoder, self).__init__()
        self.encoder = encoder_class(input_dim=obs_dim, output_dim=latent_size, hidden_dims=encoder_hidden_dims, activation=activation,last_activation=last_activation)
        # Actor network (outputs action mean)
        self.actor = MLP(obs_dim+latent_size, action_dim, hidden_dims,activation, last_activation = nn.Tanh)
        #Also learn logst of actor
        self.actor_logstd = nn.Parameter(torch.full(size=(action_dim,), fill_value=0.))
        
        self.critic = MLP(obs_dim+latent_size, 1, hidden_dims, activation)

        self.actor_optim = optim.Adam([*self.actor.parameters(), *self.encoder.parameters(), self.actor_logstd], lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

    def encode(self, obs):
        """Appends extrinsics to input observation"""
        obs_clone = obs.clone()
        obs_clone = obs_clone * torch.cat([torch.ones_like(obs_clone[..., :-1]), torch.zeros_like(obs_clone[..., -1:])], dim=-1)

        z = self.encoder(obs)

        return torch.cat([obs_clone, z], dim=-1)
    
    @torch.no_grad()
    def sample_action(self, obs):
        """Samples an action from the policy"""
        extended_obs = self.encode(obs)
        mean = self.actor(extended_obs)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        return dist.sample(), {}

    @torch.no_grad()
    def act(self, obs):
        """
            Samples an action from the actor/critic network.
        """
        ext_obs = self.encode(obs)
        mean, values = self.actor(ext_obs), self.critic(ext_obs)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(1)

        return action, log_prob, values.view(-1)
    
    @torch.no_grad()
    def get_value(self, obs):
        """Returns the critic for current observation + extrinsics"""
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
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        log_probs = dist.log_prob(acts).sum(1)

        return values.squeeze(), log_probs, dist.entropy(), true_z, encoded_z