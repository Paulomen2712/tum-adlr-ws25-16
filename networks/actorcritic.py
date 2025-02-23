import torch
from torch import nn
import torch.optim as optim
from networks.mlp import MLP
class ActorCritic(nn.Module):
    """ Actor Critic Model."""

    def __init__(self, obs_dim, action_dim, hidden_dims=[128,64], lr=1e-5, gamma=0.99, std=0., activation=nn.ELU):
        """
            Initialize parameters and build model.
        """
        super(ActorCritic, self).__init__()
        
        # Actor network (outputs action mean)
        self.actor = MLP(obs_dim, action_dim, hidden_dims,activation, last_activation = nn.Tanh)
        
        self.critic = MLP(obs_dim, 1, hidden_dims, activation)

        #Also learn logst of actor
        self.actor_logstd = nn.Parameter(torch.full(size=(action_dim,), fill_value=std))

        self.actor_optim = optim.Adam(list(self.actor.parameters()) + [self.actor_logstd], lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)


    @torch.no_grad()
    def sample_action(self, obs):
        """Samples an action from the policy"""
        mean = self.actor(obs)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        return dist.sample(), {}

    @torch.no_grad()
    def act(self, obs):
        """
            Samples an action from the actor network, it's corresponding critic and returns them along with it's log probability.
        """
        mean, values = self.actor(obs), self.critic(obs)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(1)

        return action, log_prob, values.view(-1)
    
    @torch.no_grad()
    def get_value(self, obs):
        """Returns the critic for current observation"""
        return self.critic(obs).squeeze()

    def evaluate(self, obs, acts):
        """
            Estimates the values of each observation, and the log probs of
            each action given the batch observations and actions, and the entropy. 
        """
        mean, values = self.actor(obs), self.critic(obs)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        log_probs = dist.log_prob(acts).sum(1)

        return values.flatten(), log_probs, dist.entropy()
