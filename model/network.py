import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

class Encoder(nn.Module):
    """Actor State Encoder."""

    def __init__(self, state_size, latent_size=1, fc1_units=128, fc2_units=64):
        """
            Initialize parameters and build model.

            Parameters:
                state_size (int): Dimension of each state
                latent_size (int): Dimension of the latent space
                fc1_units (int): Number of nodes in first hidden layer
                fc2_units (int): Number of nodes in second hidden layer
        """
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, latent_size)

    def forward(self, state):
        """
        Build a network that maps state -> latent representation.
        """

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class FNN(nn.Module):
    """Fully connected feedforward network with a hidden layer. """

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        """
            Initialize parameters and build model.

            Parameters:
                input_dim (int): Dimension of the input
                output_dim (int): Dimension of the output
                hidden_dim (int): Number of nodes in the hidden layer
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
    """Actor Critic Model."""

    def __init__(self, obs_dim, action_dim, hidden_dim=64, lr=1e-5, gamma=0.99):
        """
            Initialize parameters and build model.

            Parameters:
                obs_dim (int): Dimension of the observation space
                action_dim (int): Dimension of the action space
                hidden_dim (int): Number of nodes in the hidden layer
        """
        super(ActorCritic, self).__init__()
        
        # Actor network (outputs probabilities for possible actions)
        self.actor = FNN(obs_dim, action_dim, hidden_dim)
        
        # Critic network (outputs value estimate)
        self.critic = FNN(obs_dim, 1, hidden_dim)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # self.actor_scheduler= optim.lr_scheduler.StepLR(self.actor_optim, step_size = 1, gamma=gamma)
        # self.critic_scheduler= optim.lr_scheduler.StepLR(self.critic_optim, step_size = 1, gamma=gamma)
        scheduler_lambda = lambda epoch: gamma ** epoch
        self.actor_scheduler= optim.lr_scheduler.LambdaLR(self.actor_optim, lr_lambda=scheduler_lambda)
        self.critic_scheduler= optim.lr_scheduler.LambdaLR(self.critic_optim, lr_lambda=scheduler_lambda)

    
    def forward(self, obs):
        """
            Build a network that maps environment observation -> action probabilities + value estimate.
        """
        
        actions = self.actor(obs)
        critic = self.critic(obs)
        
        return actions, critic
    
    def store_savestate(self, checkpoint_path):
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
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
        self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
        print(f"Checkpoint restored from {checkpoint_path}")


class ActorCriticWithEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_size=1, hidden_dim=64, lr=1e-5, gamma=0.99, seed=0):
        """
            Initialize parameters and build model.

            Parameters:
                obs_dim (int): Dimension of the observation space
                action_dim (int): Dimension of the action space
                hidden_dim (int): Number of nodes in the hidden layer
        """
        super(ActorCriticWithEncoder, self).__init__()

        self.encoder = Encoder(obs_dim, latent_size, seed)

        # Actor network (outputs probabilities for possible actions)
        self.actor = FNN(obs_dim+latent_size, action_dim, hidden_dim)

        # Critic network (outputs value estimate)
        self.critic = FNN(obs_dim+latent_size, 1, hidden_dim)


        self.actor_optim = optim.Adam(list(self.encoder.parameters()) + list(self.actor.parameters()), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)


        scheduler_lambda = lambda epoch: gamma ** epoch
        self.actor_scheduler= optim.lr_scheduler.LambdaLR(self.actor_optim, lr_lambda=scheduler_lambda)
        self.critic_scheduler= optim.lr_scheduler.LambdaLR(self.critic_optim, lr_lambda=scheduler_lambda)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        # Ensure obs has a batch dimension
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension for single sample

        # Mask the last dimension of obs
        masked_obs = obs.clone()
        # masked_obs[..., -1] = 0
        masked_obs = masked_obs * torch.cat([torch.ones_like(masked_obs[..., :-1]), torch.zeros_like(masked_obs[..., -1:])], dim=-1)


        # Compute latent representation z
        z = self.encoder(masked_obs)

        # Concatenate z as the 10th element
        modified_obs = torch.cat([masked_obs, z], dim=-1)

        # Pass the modified_obs to actor and critic
        actions = self.actor(modified_obs).squeeze()
        critic = self.critic(modified_obs).squeeze()

        return actions, critic, modified_obs.squeeze().clone()


    def store_savestate(self, checkpoint_path):
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
