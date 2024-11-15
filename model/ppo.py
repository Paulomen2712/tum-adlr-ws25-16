import torch
from model.network import ActorCritic
from torch.distributions import MultivariateNormal
import torch.nn as nn
import numpy as np
import time
import wandb
class PPO:
    """PPO Algorithm Implementation."""
    def __init__(self, env, model = None, policy_class = ActorCritic, **hyperparameters):
        """
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class: the policy class to use for the actor/critic network.
				env: the environment to train on.
				hyperparameters: all extra PPO hyperparameters.
		"""

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # # Initialize actor and critic networks
        if model is None:
            self.policy = policy_class(self.obs_dim, self.act_dim, lr=self.lr)
        else:
            self.policy = model

        self.actor_optim = self.policy.actor_optim
        self.critic_optim = self.policy.critic_optim

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}

    def learn(self, total_timesteps):
        """
            Train the actor/critic network.

            Parameters:
                total_timesteps: the total number of timesteps to train for
        """

        t_sim = self.logger['t_so_far'] # Timesteps simulated so far
        iteration = self.logger['i_so_far']
        while t_sim < total_timesteps:   
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            t_sim += np.sum(batch_lens)
            iteration += 1

            self.logger['t_so_far'] = t_sim
            self.logger['i_so_far'] = iteration

            values, _ = self.evaluate(batch_obs, batch_acts)

            #Compute and normalise advanatage
            A_k = batch_rtgs - values.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration): 
                values, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                #Calculate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1- self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(values, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optim.step()

                self.logger['actor_losses'].append(actor_loss.detach())

            self._log_summary()

			# Save our model if it's time
            if iteration % self.save_freq == 0:
                torch.save(self.policy.state_dict(), f'./ppo_checkpoints/ppo_policy_{iteration}.pth')


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
        # Batch data. For more details.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        #Number of timesteps run so far for this batch
        t = 0
        while t < self.timesteps_per_batch:

            #Rewards of current episode
            ep_rews = []

            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t+=1

                #Store observations
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _, _ = self.env.step(action)

                #Store reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            #Store episodic length and rewards
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
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

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

        #Create Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get log_prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. 

            Parameters:
                batch_obs: the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts: the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)

            Return:
                values: the predicted values of batch_obs
                log_probs: the log probabilities of the actions taken in batch_acts given batch_obs
        """
        mean, value = self.policy(batch_obs)

        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return value.squeeze(), log_probs

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters: the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = True                              # If we should render during rollout
        self.render_every_i = 10                        # Only render every n iterations
        self.save_freq = 10                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        wandb.log({
            "simulated_timesteps": t_so_far,
            "simulated_iterations": i_so_far,
            "average_episode_lengths": avg_ep_lens,
            "average_episode_rewards": avg_ep_rews,
            "average_loss": avg_actor_loss
        })

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)