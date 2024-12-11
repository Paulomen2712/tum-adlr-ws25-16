import torch
from model.network import ActorCritic
from environments.wrappers import LunarContinuous
from torch.distributions import MultivariateNormal
import torch.nn as nn
import numpy as np
import time
import wandb
from model.storage import Storage
import os


class PPO:
    """PPO Algorithm Implementation."""

    def __init__(self, summary_writter=None, env = LunarContinuous, policy_class = ActorCritic, **hyperparameters):
        """
			Initializes the PPO model, including hyperparameters.
		"""

        # Initialize hyperparameters for training with PPO
        self.summary_writter = summary_writter
        self._init_hyperparameters(hyperparameters)

        self.storage = Storage()

        # Extract environment information
        self.env = env()
        self.obs_dim, self.act_dim =  self.env.get_environment_shape()
        
        # Initialize actor and critic
        self.policy = policy_class(self.obs_dim, self.act_dim, lr=self.lr, gamma=self.lr_gamma)
        self.actor = self.policy.actor                                              
        self.critic = self.policy.critic

        # Initialize optimizers for actor and critic
        self.actor_optim = self.policy.actor_optim  
        self.critic_optim = self.policy.critic_optim

        self.actor_scheduler = self.policy.actor_scheduler
        self.critic_scheduler = self.policy.actor_optim

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps simulated so far
			'i_so_far': 0,          # iterations simulated so far
			'batch_lens': [],       # episodic lengths in current batch
			'batch_rews': [],       # episodic returns in current batch
			'actor_losses': [],     # losses of actor network in current batch
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

            returns = advantages + self.critic(obs).squeeze().detach()
            A_k = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            batch_size = obs.size(0)
            inds = np.arange(batch_size)
            sgdbatch_size = batch_size // self.n_sgd_batches

            for _ in range(self.n_updates_per_iteration): 
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
                    V, predicted_batch_log_probs = self.evaluate(batch_obs, batch_acts)
                    ratios = torch.exp(predicted_batch_log_probs - batch_log_probs)

                    with torch.no_grad():
                        approx_kl = ((ratios - 1) - (predicted_batch_log_probs - batch_log_probs)).mean()

                    #Calculate losses
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_advantages

                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, batch_returns)

                    #Backprop
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()
                    
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                #Update learning rate
                self.actor_scheduler.step()
                self.critic_scheduler.step()

                self.logger['actor_losses'].append(actor_loss.detach())
            self.logger['lr'] = self.actor_scheduler.get_last_lr()[0]
            self._log_summary()

			# Save model every couple iterations
            if self.save_freq > 0 and iteration % self.save_freq == 0:
                save_path = self._get_save_path(iteration)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.policy.store_savestate(save_path)

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
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(self.env.reset())
        next_done = torch.zeros(self.num_envs)
        for update in range(1, self.num_updates + 1):

            for step in range(0, self.num_steps):
                global_step += 1 * self.num_envs

                with torch.no_grad():
                    actions, log_probs,value = self.get_action(next_obs)
                    values = value.flatten()

                obs, reward, done = self.env.step(actions.cpu().numpy())
                rewards = torch.tensor(reward).view(-1)
                
                advantages = self.gae(rewards, values, done)

                self.storage.add_batch(self, next_obs, actions, log_probs, rewards, values, next_done, advantages)
                next_obs, next_done = torch.Tensor(obs), torch.Tensor(done)

        return self.storage.get_rollot_data()
    
    def get_action(self, obs):
        """
            Samples an action from the actor/critic network.
        """
        mean, values = self.policy(obs)

        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach(), values.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimates the values of each observation, and the log probs of
            each action given the batch observations and actions. 
        """
        mean, values = self.policy(batch_obs)

        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return values.squeeze(), log_probs

    def gae(self, rewards, vals, dones):
        """
            Computes generalized advantage estimation (see https://arxiv.org/abs/1506.02438 page 4)
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0.0
        
        for t in reversed(range(len(vals))):
            if t + 1 < len(rewards):
                delta = rewards[t] + self.gamma * vals[t+1] * (1 - dones[t+1]) - vals[t]
            else:
                delta = rewards[t] - vals[t]

            advantages[t] = last_advantage = delta + self.gamma * self.lam * (1 - dones[t]) * last_advantage
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
                    action = self.actor(obs)
                    obs, rew, done = env.step(action.detach().numpy())

                    ep_ret += rew
                    
                val_rews.append(ep_ret)
                val_dur.append(t)
        return val_rews,  val_dur

    def test(self, env_class=LunarContinuous):
        self.policy.test()
        env = env_class(render_mode='human')
        while True:
                obs, done = env.reset()
                while not done:
                    action = self.actor(obs)
                    obs, _, done = env.step(action.detach().numpy())

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters
        """
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800                 # Number of timesteps to run per rollout
        self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update policy per iteration
        self.lr = 0.005                                 # Learning rate of policy optimizer
        self.gamma = 0.999                              # Discount factor for the rewards
        self.lam = 0.98                                 # Lambda Parameter for GAE 
        self.clip = 0.2                                 # Clip ratio for ppo loss. Using recomended 0.2
        self.max_grad_norm = 0.5                        # Gradient clipping threshold
        self.lr_gamma = 0.9998                          # Gamma for scheduler
        self.n_sgd_batches = 1                          # Number of batches for sgd

        # Misc parameters
        self.save_freq = 10                             # How often to save in number of iterations
        self.seed = None                                # Sets the seed 

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
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

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
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Current learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)