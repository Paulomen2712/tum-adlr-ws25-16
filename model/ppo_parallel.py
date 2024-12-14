import torch
from model.network import ActorCritic
from env.wrappers import LunarContinuous
from torch.distributions import MultivariateNormal
from env.wrappers import LunarContinuous
import torch.nn as nn
import numpy as np
import time
import wandb
import os


class PPO:
    """PPO Algorithm Implementation."""

    def __init__(self, summary_writter=None, env = LunarContinuous, policy_class = ActorCritic, **hyperparameters):
        """
			Initializes the PPO model, including hyperparameters.
		"""

        # Extract environment information
        self.env = env()
        self.obs_dim, self.act_dim =  self.env.get_environment_shape()

        # Extract environment information
        self.env = env()
        self.obs_dim, self.act_dim =  self.env.get_environment_shape()
         
        # Initialize hyperparameters for training with PPO
        self.summary_writter = summary_writter
        self._init_hyperparameters(hyperparameters)

        

        
        # Initialize actor and critic
        self.policy = policy_class(self.obs_dim, self.act_dim, lr=self.lr, gamma=self.lr_gamma)
        self.actor = self.policy.actor                                              
        self.critic = self.policy.critic

        # Initialize optimizers for actor and critic
        self.actor_optim = self.policy.actor_optim  
        self.critic_optim = self.policy.critic_optim

        self.actor_scheduler = self.policy.actor_scheduler
        self.critic_scheduler = self.policy.actor_optim

        self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps simulated so far
			'i_so_far': 0,          # iterations simulated so far
			'batch_lens': [],       # episodic lengths in current batch
			'batch_rews': [],       # episodic returns in current batch
			'actor_losses': [],     # losses of actor network in current batch
            'kls': [],
            'lr': self.lr           # current learning rate
		}

    def train(self, total_timesteps):
        """
            Train the actor/critic network.
        """
        self.policy.train()
        self.logger['delta_t'] = time.time_ns()
        t_sim = self.logger['t_so_far'] # Timesteps simulated so far
        iteration = self.logger['i_so_far']

        while t_sim < total_timesteps:   
            obs, acts, log_probs, ep_lens, advantages = self.rollout()

            t_sim += np.sum(ep_lens)
            iteration += 1

            self.logger['t_so_far'] = t_sim
            self.logger['i_so_far'] = iteration
            self.logger['actor_losses'] = []

            returns = advantages + self.policy.get_value(obs)
            returns = advantages + self.policy.get_value(obs).squeeze()
            A_k = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

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
                    batch_advantages = A_k[idx]
                    batch_returns = returns[idx]
                    V, predicted_batch_log_probs = self.policy.evaluate(batch_obs, batch_acts)
                    ratios = torch.exp(predicted_batch_log_probs - batch_log_probs)

                    #Calculate losses
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_advantages

                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, batch_returns)

                    #Backprop
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    if self.clip_grad:
                        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()
                    
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    if self.clip_grad:
                        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()
                    self.logger['actor_losses'].append(actor_loss.detach())
                    batch_values, pred_batch_log_probs = self.policy.evaluate(batch_obs, batch_acts)
                    
                    actor_loss, kl = self.update_actor(pred_batch_log_probs, batch_log_probs, batch_advantages)
                    self.update_critic(batch_values, batch_returns)

                    #Update learning rate
                    # self.actor_scheduler.step()
                    # self.critic_scheduler.step()
                    loss.append(actor_loss.detach())
                    kls.append(kl)

                self.logger['actor_losses'].append(np.mean(loss))
                self.logger['kls'].append(np.mean(kls))
            self.logger['lr'] = self.actor_scheduler.get_last_lr()[0]
            self._log_summary()

			# Save model every couple iterations
            if self.save_freq > 0 and iteration % self.save_freq == 0:
                save_path = self._get_save_path(iteration)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.policy.store_savestate(save_path)

        #t

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
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []
        batch_advantages = []

        ep_rews = []
        ep_vals = []
        ep_dones = []

        t = 0
        while t < self.timesteps_per_batch:

            ep_rews = []
            ep_vals = []
            ep_dones = []

            obs, done = self.env.reset()

            for ep_t in range(self.max_timesteps_per_episode):
                t+=1

                batch_obs.append(obs)
                ep_dones.append(done)

                action, log_prob, val = self.policy.act(obs)
                action, log_prob, val = self.policy.act(obs)
                obs, rew, done = self.env.step(action)

                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            batch_lens.append(ep_t + 1)
            batch_rews.append(np.sum(ep_rews))
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
            batch_advantages.extend(self.gae(ep_rews, ep_vals, ep_dones))

        batch_advantages = torch.tensor(batch_advantages, dtype=torch.float)
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_lens, batch_advantages
    
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
        critic_loss = nn.MSELoss()(values, returns)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()

    def gae(self, rewards, vals, dones):
        """
            Computes generalized advantage estimation (see https://arxiv.org/abs/1506.02438 page 4)
        """
        advantages = []
        last_advantage = 0.0
        
        for t in reversed(range(len(vals))):
            if t + 1 < len(rewards):
                delta = rewards[t] + self.gamma * vals[t+1] * (1 - dones[t+1]) - vals[t]
            else:
                delta = rewards[t] - vals[t]

            advantage = delta + self.gamma * self.lam * (1 - dones[t]) * last_advantage
            last_advantage = advantage
            advantages.insert(0, advantage)
        return advantages

    def restore_savestate(self, checkpoint):
        model = ActorCritic(self.obs_dim, self.act_dim)
        model.restore_savestate(checkpoint)
        self.policy = model

    def validate(self, max_iter, should_record=False, env_class=LunarContinuous):
        self.policy.eval()
        if should_record:
            env = env_class()
            env.make_environment_for_recording()
        else:
            env = env_class(render_mode = 'human')
        val_rews = []
        val_dur = []
        for _ in range(0, max_iter) :
                obs, done = env.reset()

                t = 0
                ep_ret = 0

                while not done:
                    t += 1
                    action = self.policy.sample_action(obs)
                    obs, rew, done = env.step(action)

                    ep_ret += rew
                    
                val_rews.append(ep_ret)
                val_dur.append(t)
        return val_rews,  val_dur

    def test(self, env_class=LunarContinuous):
        self.policy.eval()
        env = env_class(render_mode='human')
        while True:
                obs, done = env.reset()
                while not done:
                    action = self.policy.sample_action(obs)
                    obs, _, done = env.step(action)

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters
        """
        for param, val in self.env.load_hyperparameters().items():
            setattr(self, param, val)

        config_hyperparameters = self.env.load_hyperparameters()
        for param, val in config_hyperparameters.items():
            setattr(self, param, val)

        for param, val in hyperparameters.items():
            setattr(self, param, val)

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
        """
            Print to stdout the results for the most recent batch. Additionaly log data to wandb if applicable.
        """
        lr = self.logger['lr']
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean(self.logger['batch_rews'])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean(self.logger['actor_losses'])
        avg_kl = np.mean(self.logger['kls'])

        #log to wandb
        if self.summary_writter is not None:
            self.summary_writter.save_dict({
                "simulated_timesteps": t_so_far,
                "simulated_iterations": i_so_far,
                "average_episode_lengths": avg_ep_lens,
                "average_episode_rewards": avg_ep_rews,
                "average_loss": avg_actor_loss,
                "learning_rate": lr
            })

        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Average KL: {avg_kl}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Current learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)