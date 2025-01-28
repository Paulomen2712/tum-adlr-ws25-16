import torch
from torch import nn
import torch.optim as optim
from networks.mlp import MLP
import copy
from collections import deque
from networks.policy import AdaptivePolicy
from torch.distributions import MultivariateNormal
class AdaptiveActorCritic(AdaptivePolicy):
    """ Actor Critic Model."""

    def __init__(self, obs_dim, action_dim, latent_size=1, encoder_hidden_dims=[64, 32], lr=1e-5, history_len=10, activation=nn.ELU):
        """
            Initialize parameters and build model.
        """
        
        super(AdaptiveActorCritic, self).__init__(obs_dim, action_dim, latent_size, encoder_hidden_dims)

        self.history_len = history_len
        self.obs_history = deque(maxlen=history_len)
        self.action_history = deque(maxlen=history_len)

        encoder_input_dim = (obs_dim * history_len) + (action_dim * history_len)
        self.encoder = MLP(encoder_input_dim, latent_size, encoder_hidden_dims, activation=activation)

        self.optim = optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor = None
        self.actor_logstd = None

    def clear_history(self):
        self.obs_history.clear()
        self.action_history.clear()

    def set_policy(self, policy):
        self.actor = copy.deepcopy(policy.actor)
        for param in self.actor.parameters():
            param.requires_grad = False
        self.actor_logstd = copy.deepcopy(policy.actor_logstd)
        self.actor_logstd.requires_grad = False

    def encode(self, obs):
        # Ensure the observation has the same shape as the last one in the history, or pad/trim as necessary
        # print(f"input obs shape: {obs.shape}")
        if len(self.obs_history) > 0:
            expected_size = self.obs_history[-1].shape[0]
            
            current_size = obs.shape[0]

            if current_size < expected_size:
                padding = torch.zeros((expected_size - current_size, obs.shape[1]), device=obs.device)
                obs = torch.cat([obs, padding], dim=0)
            elif current_size > expected_size:
                obs = obs[..., :expected_size]
        

        masked_obs = obs.clone()
        masked_obs = masked_obs * torch.cat([torch.ones_like(masked_obs[..., :-1]), torch.zeros_like(masked_obs[..., -1:])], dim=-1)

        # Append the masked observation to the history
        self.obs_history.append(masked_obs.clone())


        # Handling action history
        if len(self.action_history) < self.history_len:
            if len(self.action_history) > 0:
                padded_actions = [self.action_history[0].clone()] * (self.history_len - len(self.action_history))
            else:
                # TODO: Hard-coded action_dim in padding
                padded_actions = [torch.zeros((obs.shape[0], 2), device=obs.device)] * self.history_len
            all_actions = padded_actions + list(self.action_history)
        else:
            all_actions = list(self.action_history)

        # Handling observation history
        if len(self.obs_history) < self.history_len:
            if len(self.obs_history) > 0:
                padded_obs = [torch.zeros_like(obs)] * (self.history_len - len(self.obs_history))
            else:
                padded_obs = [torch.zeros_like(obs)] * self.history_len
            all_obs = padded_obs + list(self.obs_history)
        else:
            all_obs = list(self.obs_history)

        # print([obss.shape for obss in all_obs])
        # Ensure all observations in the history have the same shape before concatenating
        all_obs_tensor = torch.cat(all_obs, dim=-1)
        
        # Ensure all actions in the history have the same shape before concatenating
        all_actions_tensor = torch.cat(all_actions, dim=-1)

        # Combine actions and observations into one history tensor
        history_tensor = torch.cat([all_obs_tensor, all_actions_tensor], dim=-1)

        # Encode the history tensor
        z = self.encoder(history_tensor)

        # Return the concatenated masked observation and encoded history
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

    def evaluate(self, obs):
        """
            Estimates the values of each observation, and the log probs of
            each action given the batch observations and actions. 
        """
        return self.encode(obs)[:,-1]

    # def evaluate(self, obs):
    #     """
    #         Estimates the values of each observation, and the log probs of
    #         each action given the batch observations and actions. 
    #     """
    #     ext_obs = self.encode(obs)
    #     mean = self.actor(ext_obs)
    #     action_std = torch.exp(self.actor_logstd)
    #     dist = torch.distributions.Normal(mean, action_std)
    #     action = dist.sample()
    #     print(action)

    #     self.action_history.append(action.clone())

    #     return action.squeeze()
    # def evaluate(self, obs):
    #     """
    #     Passes each observation one by one to populate history and returns a tensor of values (batch_size x 1).
    #     """
    #     batch_size = obs.shape[0]
    #     values = []

    #     for i in range(batch_size):
    #         # Process one observation at a time, adding a batch dimension
    #         current_obs = obs[i].unsqueeze(0)  # Shape: [1, obs_dim]
    #         encoded_obs = self.encode(current_obs)  # Shape: [1, new_obs_dim]

    #         mean = self.actor(encoded_obs)
    #         action_std = torch.exp(self.actor_logstd)
    #         dist = torch.distributions.Normal(mean, action_std)
    #         action = dist.sample()

            

    #         # Extract the last value for the current observation
    #         value = encoded_obs[0, -1]  # Access the first (and only) batch dimension
    #         values.append(value)

    #     # Stack all values into a single tensor (batch_size x 1)
    #     values_tensor = torch.stack(values, dim=0).unsqueeze(1)  # Shape: [batch_size, 1]

    #     return values_tensor
    
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