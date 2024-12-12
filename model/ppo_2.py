import torch
from model.network import ActorCritic
from env.wrappers import LunarContinuous
import torch.nn as nn
import numpy as np
import time
import wandb
import os
from utils.storage import Storage


class PPO:
    """PPO Algorithm Implementation."""

    def __init__(self, summary_writter=None, env = LunarContinuous, policy_class = ActorCritic, **hyperparameters):
        """
			Initializes the PPO model, including hyperparameters.
		"""
        # Extract environment information
        self.env = env()
        self.obs_dim, self.act_dim =  self.env.get_environment_shape()
         
        # Initialize hyperparameters for training with PPO
        self.summary_writter = summary_writter
        self._init_hyperparameters(hyperparameters)

        self.storage = Storage(self.num_steps, self.num_envs, self.obs_dim, self.act_dim, self.gamma, self.lam)

        # Initialize actor and critic
        self.policy = policy_class(self.obs_dim, self.act_dim, lr=self.lr, gamma=self.lr_gamma)#.to(self.device)
        self.actor = self.policy.actor                                              
        self.critic = self.policy.critic

        # Initialize optimizers for actor and critic
        self.actor_optim = self.policy.actor_optim  
        self.critic_optim = self.policy.critic_optim

        self.actor_scheduler = self.policy.actor_scheduler
        self.critic_scheduler = self.policy.actor_optim

        self.logger = {
			'delta_t': time.time_ns(),
			'i_so_far': 0,          # iterations simulated so far
			'batch_rews': 0,       # episodic returns in current batch
			'actor_losses': [],     # losses of actor network in current batch
            'kls': [],
            'lr': self.lr           # current learning rate
		}

    def train(self):
        """
            Train the actor/critic network.
        """
        self.policy.train()
        self.logger['delta_t'] = time.time_ns()

        for it in range(0, self.base_train_it):   
            obs, acts, log_probs, advantages, returns = self.rollout()

            self.logger['i_so_far'] = it + 1
            self.logger['batch_rews'] = self.storage.get_average_episode_rewards()


            batch_size = obs.size(0)
            inds = np.arange(batch_size)
            sgdbatch_size = batch_size // self.n_sgd_batches
            loss = []
            kls = []

            for _ in range(self.n_updates_per_iteration): 

                # frac = (t_sim - 1.0) / total_timesteps
                # new_lr = self.lr * (1.0 - frac)

                # new_lr = max(new_lr, 0.0)
                # self.actor_optim.param_groups[0]["lr"] = new_lr
                # self.critic_optim.param_groups[0]["lr"] = new_lr
                # # Log learning rate
                # self.logger['lr'] = new_lr

                #SGD
                np.random.shuffle(inds)
                for start in range(0, batch_size, sgdbatch_size):
                    end = start + sgdbatch_size
                    idx = inds[start:end]
                    
                    #Restrict data to current batch
                    batch_obs = obs[idx]
                    batch_acts = acts[idx]
                    batch_log_probs = log_probs[idx]
                    batch_advantages = advantages[idx]
                    batch_returns = returns[idx]
                    batch_values, pred_batch_log_probs = self.policy.evaluate(batch_obs, batch_acts)
                    
                    actor_loss, kl = self.update_actor(pred_batch_log_probs, batch_log_probs, batch_advantages)
                    self.update_critic(batch_values, batch_returns)

                    #Update learning rate
                    # self.actor_scheduler.step()
                    # self.critic_scheduler.step()
                    loss.append(actor_loss.detach().item())
                    kls.append(kl.item())

                self.logger['actor_losses'].append(np.mean(loss))
                self.logger['kls'].append(np.mean(kls))
            self.logger['lr'] = self.actor_scheduler.get_last_lr()[0]
            self._log_summary()

			# Save model every couple iterations
            if self.save_freq > 0 and it+1 % self.save_freq == 0:
                save_path = self._get_save_path(it+1)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.policy.store_savestate(save_path)

    def rollout2(self):
        """
            Collects batch of simulated data.

            Return:
                batch_obs: the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts: the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs: the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_lens: the lengths of each episode this batch. Shape: (number of episodes)
                batch_advantages: the advantages collected from this batchShape: (number of timesteps)
        """

        obs = torch.zeros((self.num_steps, self.num_envs, self.obs_dim))
        actions = torch.zeros((self.num_steps, self.num_envs, self.act_dim))
        logprobs = torch.zeros((self.num_steps, self.num_envs))
        rewards = torch.zeros((self.num_steps, self.num_envs))
        dones = torch.zeros((self.num_steps, self.num_envs))
        values = torch.zeros((self.num_steps, self.num_envs))

        next_obs, next_done = self.env.reset()
        next_obs, next_done = torch.Tensor(next_obs), torch.Tensor(next_done)
        

        for step in range(self.num_steps):
            obs[step] = next_obs
            dones[step] = next_done

            action, logprob, val = self.policy.act(next_obs)
            next_obs, rew, next_done = self.env.step(action.numpy())


            values[step] = val
            actions[step] = action
            logprobs[step] = logprob
            rewards[step] = torch.Tensor(rew)
            next_obs, next_done = torch.Tensor(next_obs), torch.Tensor(next_done)
        
        next_value = self.policy.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards)
        last_lam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_lam = delta + self.gamma * self.lam * next_non_terminal * last_lam

        returns = advantages + values
        if self.norm_adv:
            advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        print(self.get_average_episode_rewards(rewards, dones))
        return self.format_tensor(obs).reshape(-1, self.obs_dim), self.format_tensor(actions).reshape(-1, self.act_dim), self.format_tensor(logprobs).flatten(), self.format_tensor(values).flatten(), self.format_tensor(advantages).flatten(), self.format_tensor(returns).flatten(), rewards[dones == 0].sum(dim=0).mean()

    def rollout(self):
        """
            Collects batch of simulated data.

            Return:
                batch_obs: the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts: the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs: the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_lens: the lengths of each episode this batch. Shape: (number of episodes)
                batch_advantages: the advantages collected from this batchShape: (number of timesteps)
        """
        self.storage.clear()

        next_obs, next_done = self.env.reset()
        for _ in range(self.num_steps):
            obs = next_obs
            dones = next_done

            actions, logprobs, values = self.policy.act(torch.from_numpy(next_obs))
            next_obs, rewards, next_done = self.env.step(actions.numpy())

            self.storage.store_batch(obs, actions, logprobs, rewards, values, dones)
        
        self.storage.compute_advantages(self.policy.get_value(torch.from_numpy(obs)), next_done)
        return self.storage.get_rollot_data()

    def format_tensor(self, tensor):
        return tensor.transpose(0,1)

    def update_actor(self, pred_log_probs, log_probs, advantages):
        log_ratios = pred_log_probs - log_probs
        ratios = torch.exp(log_ratios)
        surr1 = -ratios * advantages
        surr2 = -torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
        with torch.no_grad():
            kl = ((ratios - 1) - log_ratios).mean()
        actor_loss = (torch.max(surr1, surr2)).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()
        return actor_loss.detach(), kl

    def update_critic(self,  values, returns):
        critic_loss = nn.MSELoss()(values.squeeze(), returns.squeeze())
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()

    def restore_savestate(self, checkpoint):
        model = ActorCritic(self.obs_dim, self.act_dim)
        model.restore_savestate(checkpoint)
        self.policy = model

    def validate(self, max_iter, should_record=False, env_class=LunarContinuous):
        self.policy.eval()
        if should_record:
            env = env_class(num_envs=max_iter,should_record='True')
        else:
            env = env_class(num_envs=max_iter)
        val_rews = []
        val_dur = []
        obs, done  = env.reset()

        t = np.array([0]*max_iter)
        ep_ret = np.array([0.]*max_iter)

        while not all(done):
            not_done = np.array([1]*max_iter) - done
            t += not_done
            action = self.actor(torch.Tensor(obs))
            obs, rew, next_done = env.step(action.detach().numpy())
            done |= next_done
            ep_ret += rew * not_done
            
        val_rews.append(ep_ret)
        val_dur.append(t)
        return val_rews,  val_dur

    def test(self, env_class=LunarContinuous):
        self.policy.eval()
        env = env_class(num_envs=1,render_mode='human')
        while True:
                obs, done = env.reset()
                print(obs, flush=True)
                while not done[0]:
                    action = self.actor(torch.Tensor(obs))
                    obs, _, done = env.step(action.detach().numpy())

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters
        """

        config_hyperparameters = self.env.load_hyperparameters()
        for param, val in config_hyperparameters.items():
            setattr(self, param, val)

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        if self.seed != None:
            assert(type(self.seed) == int)

            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _get_save_path(self, iteration):
        if self.summary_writter is None:
            return f'./ppo_checkpoints/non_wandb/ppo_policy_{iteration}.pth'
        else:
            return f'./ppo_checkpoints/{wandb.run.name}/ppo_policy_{iteration}.pth'

    def _log_summary(self):
        lr = self.logger['lr']
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        i_so_far = self.logger['i_so_far']
        avg_ep_rews = self.logger['batch_rews'].item()
        avg_actor_loss = np.mean([losses for losses in self.logger['actor_losses']])
        avg_kl = np.mean([kl for kl in self.logger['kls']])

        #log to wandb
        if self.summary_writter is not None:
            self.summary_writter.save_dict({
                "simulated_iterations": i_so_far,
                "average_episode_rewards": avg_ep_rews,
                "average_loss": avg_actor_loss,
                "learning_rate": lr
            })

        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Average KL Divergence: {avg_kl}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Current learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)