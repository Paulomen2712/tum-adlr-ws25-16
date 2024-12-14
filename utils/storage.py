import torch
import numpy as np
class Storage():
    def __init__(self, num_steps, num_envs, obs_dim, act_dim, gamma, lam, device='cuda'):
        self.device = torch.device(device)
        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=self.device)
        self.actions = torch.zeros((num_steps, num_envs, act_dim), device=self.device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=self.device)
        self.rewards = torch.zeros((num_steps, num_envs), device=self.device)
        self.dones = torch.zeros((num_steps, num_envs), device=self.device)
        self.values = torch.zeros((num_steps, num_envs), device=self.device)
        self.returns = torch.zeros((num_steps, num_envs), device=self.device)
        self.advantages = torch.zeros((num_steps, num_envs), device=self.device)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_envs = num_envs

        self.gamma = gamma
        self.lam = lam

        self.num_steps = num_steps
        self.step = 0
        self.device = device

    def store_batch(self, obs, actions, logprobs, rewards, values, dones):
        if self.step >= self.num_steps:
            raise AssertionError("Rollout buffer overflow")
        self.obs[self.step].copy_(torch.from_numpy(obs).to(self.device))
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(torch.from_numpy(rewards).to(self.device))
        self.dones[self.step].copy_(torch.from_numpy(dones).to(self.device))
        self.values[self.step].copy_(values)
        self.logprobs[self.step].copy_(logprobs.to(self.device))
        self.step += 1

    def compute_advantages(self, next_value, next_done):
        next_value = next_value.reshape(1, -1)
        next_done = torch.Tensor(next_done).to(self.device)
        last_lam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_lam = delta + self.gamma * self.lam * next_non_terminal * last_lam

    def clear(self):
        self.step = 0
    
    def get_average_episode_rewards(self):
        # May not 100% represent the full episode rewards, but gives a good idea. Computes the rewards of the first finished episode per environment
        # For more accurate results please validate the model with ppo.validate
        done_indices = torch.argmax(self.dones, dim=0)
        done_indices = torch.where(self.dones.any(dim=0), done_indices, self.num_steps)
        mask = torch.arange(self.num_steps, device=self.device).unsqueeze(1) <= done_indices.unsqueeze(0)
        masked_rewards = self.rewards * mask.float()
        sum_rewards = masked_rewards.sum(dim=0)

        return sum_rewards.mean().cpu()
    
    def get_rollot_data(self):
        obs = self.obs.transpose(0,1).reshape((-1,self.obs_dim))
        logprobs = self.logprobs.transpose(0,1).flatten()
        actions = self.actions.transpose(0,1).reshape((-1, self.act_dim))
        returns = (self.advantages + self.values).transpose(0,1).flatten()
        advantages = self.advantages.transpose(0,1).flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        return  obs, actions, logprobs, advantages, returns
     
    def store_values( self, values):
        self.values[self.step].copy_(values)
        step+=1
    def get_values(self):
        return self.values.transpose(0,1).flatten()