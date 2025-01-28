import torch
from torch import nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from networks.mlp import MLP
class ActorCritic(nn.Module):
    """ Actor Critic Model."""

    def __init__(self, obs_dim, action_dim, hidden_dims=[128,64], lr=1e-5, gamma=0.99, std=0., activation=nn.ELU):
        """
            Initialize parameters and build model.
        """
        super(ActorCritic, self).__init__()
        
        # Actor network (outputs probabilities for possible actions)
        self.actor = MLP(obs_dim, action_dim, hidden_dims,activation, last_activation = nn.Tanh)
        
        # Critic network (outputs value estimate)
        self.critic = MLP(obs_dim, 1, hidden_dims, activation)

        self.actor_logstd = nn.Parameter(torch.full(size=(action_dim,), fill_value=std))

        self.actor_optim = optim.Adam(list(self.actor.parameters()) + [self.actor_logstd], lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # self.actor_scheduler= optim.lr_scheduler.StepLR(self.actor_optim, step_size = 1, gamma=gamma)
        # self.critic_scheduler= optim.lr_scheduler.StepLR(self.critic_optim, step_size = 1, gamma=gamma)
        scheduler_lambda = lambda epoch: gamma ** epoch
        self.actor_scheduler= optim.lr_scheduler.LambdaLR(self.actor_optim, lr_lambda=scheduler_lambda)
        self.critic_scheduler= optim.lr_scheduler.LambdaLR(self.critic_optim, lr_lambda=scheduler_lambda)


    @torch.no_grad()
    def sample_action(self, obs):
        mean = self.actor(obs)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        return dist.sample()

    @torch.no_grad()
    def act(self, obs):
        """
            Samples an action from the actor/critic network.
        """
        mean, values = self.actor(obs), self.critic(obs)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(1)

        return action, log_prob, values.view(-1)
    
    @torch.no_grad()
    def get_value(self, obs):
        return self.critic(obs).squeeze()

    def evaluate(self, obs, acts):
        """
            Estimates the values of each observation, and the log probs of
            each action given the batch observations and actions. 
        """
        mean, values = self.actor(obs), self.critic(obs)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, action_std)
        log_probs = dist.log_prob(acts).sum(1)

        return values.flatten(), log_probs, dist.entropy()
