import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.distributions import MultivariateNormal

class Encoder(nn.Module):
    """ Actor State Encoder. """

    def __init__(self, state_size, latent_size, seed, fc1_units=128, fc2_units=64):
        """
            Initialize parameters and build model.

        """
        super(Encoder, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, latent_size)

    def forward(self, state):
        """
            Build a network that maps state -> latent representation.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class FNN(nn.Module):
    """ Fully connected feedforward network with a hidden layer. """

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        """
            Initialize parameters and build model.
        """
        super(FNN, self).__init__()
		
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.fc1(obs))
        activation2 = F.relu(self.fc2(activation1))
        output = self.fc3(activation2)

        return output
    
class ActorCritic(nn.Module):
    """ Actor Critic Model."""

    def __init__(self, obs_dim, action_dim, hidden_dim=64, lr=1e-5, gamma=0.99, std=0.5):
        """
            Initialize parameters and build model.
        """
        super(ActorCritic, self).__init__()
        
        # Actor network (outputs probabilities for possible actions)
        self.actor = FNN(obs_dim, action_dim, hidden_dim)
        self.cov_var = torch.full(size=(action_dim,), fill_value=std)
        self.cov_mat = torch.diag(self.cov_var)
        
        # Critic network (outputs value estimate)
        self.critic = FNN(obs_dim, 1, hidden_dim)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # self.actor_scheduler= optim.lr_scheduler.StepLR(self.actor_optim, step_size = 1, gamma=gamma)
        # self.critic_scheduler= optim.lr_scheduler.StepLR(self.critic_optim, step_size = 1, gamma=gamma)
        scheduler_lambda = lambda epoch: gamma ** epoch
        self.actor_scheduler= optim.lr_scheduler.LambdaLR(self.actor_optim, lr_lambda=scheduler_lambda)
        self.critic_scheduler= optim.lr_scheduler.LambdaLR(self.critic_optim, lr_lambda=scheduler_lambda)

    
    # def forward(self, obs):
    #     """
    #         Build a network that maps environment observation -> action probabilities + value estimate.
    #     """
        
    #     actions = self.actor(obs)
    #     critic = self.critic(obs)
        
    #     return actions, critic

    @torch.no_grad()
    def get_value(self, obs):
        return self.critic(obs).squeeze()

    @torch.no_grad()
    def act(self, obs):
        """
            Samples an action from the actor/critic network.
        """
        mean, values = self.actor(obs), self.critic(obs)

        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.numpy(), log_prob, values
    
    @torch.no_grad()
    def sample_action(self, obs):
        mean= self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()

        return action.numpy()

    def evaluate(self, obs, acts):
        """
            Estimates the values of each observation, and the log probs of
            each action given the batch observations and actions. 
        """
        mean, values = self.actor(obs), self.critic(obs)

        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(acts)

        return values.squeeze(), log_probs
    
    def store_savestate(self, checkpoint_path):
        """
            Stores the model into the given directory.
        """
        checkpoint = {
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
        """
            Loads the model from the given directory.
        """
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
        self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
        print(f"Checkpoint restored from {checkpoint_path}")
