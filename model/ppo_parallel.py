import torch
from model.network import ActorCritic
from model.environments import LunarContinuous
from torch.distributions import MultivariateNormal
import torch.nn as nn
import numpy as np
import time
import wandb
from multiprocessing import Queue, Process
import os


class PPO:
    """PPO Algorithm Implementation."""

    def __init__(self, env_mker = LunarContinuous(), model = None, policy_class = ActorCritic, **hyperparameters):
        """
			Initializes the PPO model, including hyperparameters.

			Parameters:
                env_mker: the class the creates the environment to train on
				model: if should continue training on a pre-existing model
                policy_class: the policy class to use for the actor/critic network
				hyperparameters: all extra hyperparameters
		"""

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env_mker = env_mker
        self.env = self.env_mker.make_environment()
        self.obs_dim =  self.env.observation_space.shape[0]
        self.act_dim =  self.env.action_space.shape[0]

        # Initialize actor and critic networks
        if model is None:
            self.policy = policy_class(self.obs_dim, self.act_dim, lr=self.lr, gamma=self.lr_gamma)
        else:
            self.policy = model

        # Initialize actor and critic
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
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
            'lr': self.lr
		}

    def learn(self, total_timesteps):
        """
            Train the actor/critic network.

            Parameters:
                total_timesteps: the total number of timesteps to train for
        """
        self.logger['delta_t'] = time.time_ns() # Reset time to allow multiple learning episodes
        t_sim = self.logger['t_so_far'] # Timesteps simulated so far
        iteration = self.logger['i_so_far']

        while t_sim < total_timesteps:   
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            t_sim += np.sum(batch_lens)
            iteration += 1

            self.logger['t_so_far'] = t_sim
            self.logger['i_so_far'] = iteration

            V, _ = self.evaluate(batch_obs, batch_acts)

            #Advanatage calculation
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration): 
                # frac = (t_sim - 1.0) / total_timesteps
                # new_lr = self.lr * (1.0 - frac)
                

                # # Make sure learning rate doesn't go below 0
                # new_lr = max(new_lr, 0.0)
                # self.logger['lr'] = new_lr
                # self.actor_optim.param_groups[0]["lr"] = new_lr
                # self.critic_optim.param_groups[0]["lr"] = new_lr

                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                #Calculate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1- self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                #Backpropagate
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()
                self.actor_scheduler.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                self.critic_scheduler.step()

                self.logger['actor_losses'].append(actor_loss.detach())
            self.logger['lr'] = self.actor_scheduler.get_last_lr()[0]
            # self.actor_scheduler.step()
            # self.critic_scheduler.step()
            self._log_summary()

			# Save model every couple iterations
            if iteration % self.save_freq == 0:
                save_path = f'./ppo_parallel_checkpoints/{wandb.run.name}/ppo_policy_{iteration}.pth'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.policy.state_dict(), save_path)

    def rollout(self):
        """
            Collects batch of simulated data.

            Return:
                batch_obs: the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts: the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs: the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs: the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens: the lengths of each episode this batch. Shape: (number of episodes)
        """
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        t = 0
        while t < self.timesteps_per_batch:

            ep_rews = []

            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t+=1

                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _, _ = self.env.step(action)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)     

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
      
    def compute_rtgs(self, batch_rews):
        """
			Compute the Reward-To-Go of each timestep given the reawards of the batch.

			Parameters:
				batch_rews: the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs: the rewards to go, Shape: (number of timesteps in batch)
		"""
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_rew = 0

            for rew in reversed(ep_rews):
                discounted_rew = rew + discounted_rew*self.gamma
                batch_rtgs.insert(0, discounted_rew)
        
        return torch.tensor(np.array(batch_rtgs), dtype=torch.float)
    
    def get_action(self, obs):
        """
            Queries an action from the actor/critic network.

			Parameters:
				obs: the observation at the current timestep

			Return:
				action: the action to take, as a numpy array
				log_prob: the log probability of the selected action in the distribution
        """
        mean, _ = self.policy(obs)

        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor/critic network. 

            Parameters:
                batch_obs: the observations from the most recently collected batch as a tensor of
                            shape: (number of timesteps in batch, dimension of observation)
                batch_acts: the actions from the most recently collected batch as a tensor of
                            shape: (number of timesteps in batch, dimension of action)

            Return:
                values: the predicted values of batch_obs
                log_probs: the log probabilities of the actions taken in batch_acts given batch_obs
        """
        mean, values = self.policy(batch_obs)

        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return values.squeeze(), log_probs

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters: the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values
        """
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.lr_gamma = 0.99

        # Miscellaneous parameters
        self.save_freq = 10                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results
        self.num_workers = 8                            # Sets the ammount of workers to parallelise rollouts
        self.post_results = True                        # Whether to log the results to wanb

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        if self.seed != None:
            # Check if seed is valid first
            assert(type(self.seed) == int)

            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch. Additionaly log data to wandb if flag is set.
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
        if self.post_results:
            wandb.log({
                "simulated_timesteps": t_so_far,
                "simulated_iterations": i_so_far,
                "average_episode_lengths": avg_ep_lens,
                "average_episode_rewards": avg_ep_rews,
                "average_loss": avg_actor_loss,
                "learning_rate": lr
            })

        # Round decimal places
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