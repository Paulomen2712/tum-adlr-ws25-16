import torch
from model.network import ActorCritic
from model.environments import LunarContinuous
from torch.distributions import MultivariateNormal
import torch.nn as nn
import numpy as np
import time
import wandb
from multiprocessing import Pool
import os


class PPO:
    """PPO Algorithm Implementation."""

    def __init__(self, summary_writter=None, env_mker = LunarContinuous(), policy_class = ActorCritic, **hyperparameters):
        """
			Initializes the PPO model, including hyperparameters.

			Parameters:
                env_mker: the class the creates the environment to train on
				model: if should continue training on a pre-existing model
                policy_class: the policy class to use for the actor/critic network
				hyperparameters: all extra hyperparameters
		"""

        # Initialize hyperparameters for training with PPO
        self.summary_writter = summary_writter
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env_mker = env_mker
        self.env = self.env_mker.make_environment()
        self.obs_dim =  self.env.observation_space.shape[0]
        self.act_dim =  self.env.action_space.shape[0]

        
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
			't_so_far': 0,             # timesteps simulated so far
			'i_so_far': 0,             # iterations simulated so far
			'batch_lens': [],          # episodic lengths in current batch
			'batch_rews': [],          # episodic returns in current batch
			'actor_losses': [],        # losses of actor network in current batch
            'lr': self.lr,             # current learning rate
            'avg_rollout_t': 0 #average runtime of individual rollout parallelisations
		}

    def train(self, total_timesteps):
        """
            Train the actor/critic network.

            Parameters:
                total_timesteps: the total number of timesteps to train for
        """
        self.policy.train()
        self.logger['delta_t'] = time.time_ns()
        t_sim = self.logger['t_so_far'] # Timesteps simulated so far
        iteration = self.logger['i_so_far']

        while t_sim < total_timesteps:   
            obs, acts, log_probs, rews, ep_lens, vals, dones = self.rollout()

            t_sim += np.sum(ep_lens)
            iteration += 1

            self.logger['t_so_far'] = t_sim
            self.logger['i_so_far'] = iteration

            V = self.critic(obs).squeeze()

            #Advanatage calculation
            A_k, returns = self.compute_gae(obs, rews, vals, dones)

            batch_size = obs.size(0)
            inds = np.arange(batch_size)
            sgdbatch_size = batch_size // self.num_minibatches
            loss = []

            for _ in range(self.n_updates_per_iteration): 

                np.random.shuffle(inds)
                #SGD
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

                    #Calculate losses
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_advantages

                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, batch_returns)

                    #Backpropagate
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()
                    
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                self.actor_scheduler.step()
                self.critic_scheduler.step()

                self.logger['actor_losses'].append(actor_loss.detach())
            self.logger['lr'] = self.actor_scheduler.get_last_lr()[0]
            self._log_summary()

			# Save model every couple iterations
            if iteration % self.save_freq == 0:
                save_path = f'./ppo_parallel_checkpoints/{wandb.run.name}/ppo_policy_{iteration}.pth'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.policy.state_dict(), save_path)

    def worker_rollout(self, _):
        """
        Worker function that collects a single environment's rollout data.
        Measures and returns the duration of the rollout in addition to the data.
        Returns:
            A dictionary containing observations, actions, log_probs, rewards, values, dones, episode length,
            and the time taken for this worker's rollout.
        """
        start_time = time.time()
        env = self.env_mker.make_environment()
        obs, _ = env.reset()
        done = False

        env_obs = []
        env_acts = []
        env_log_probs = []
        env_rews = []
        env_vals = []
        env_dones = []

        for _ in range(self.max_timesteps_per_episode):
            env_obs.append(obs)
            env_dones.append(done)

            action, log_prob = self.get_action(obs)
            val = self.critic(obs)
            obs, rew, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            env_rews.append(rew)
            env_vals.append(val.detach().flatten())
            env_acts.append(action)
            env_log_probs.append(log_prob)

            if done:
                break

        env.close()
        duration = time.time() - start_time

        return {
            'obs': env_obs,
            'acts': env_acts,
            'log_probs': env_log_probs,
            'rewards': env_rews,
            'values': env_vals,
            'dones': env_dones,
            'length': len(env_rews),
            'duration': duration
        }

    def rollout(self):
        """
            Collects batch of simulated data.

            Return:
                batch_obs: the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts: the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs: the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_lens: the lengths of each episode this batch. Shape: (number of episodes)
        """
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []

        rollout_durations = []

        total_timesteps = 0
        with Pool(processes=self.num_workers) as pool:
            while total_timesteps < self.timesteps_per_batch:

                results = pool.map(self.worker_rollout, range(self.num_workers))

                for result in results:
                    batch_obs.extend(result['obs'])
                    batch_acts.extend(result['acts'])
                    batch_log_probs.extend(result['log_probs'])
                    batch_rews.append(result['rewards'])
                    batch_lens.append(result['length'])
                    batch_vals.append(result['values'])
                    batch_dones.append(result['dones'])
                    rollout_durations.append(result['duration'])
                    total_timesteps += result['length']

                    if total_timesteps >= self.timesteps_per_batch:
                        break

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones
    
    def get_action(self, obs):
        """
            Samples an action from the actor/critic network.

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

    def compute_gae(self,  obs, rewards, values, dones):
        advantages = []

        for ep_rewards, ep_vals, ep_dones in zip(rewards, values, dones):
            ep_advantages = []
            last_advantage = 0.0
            
            for t in reversed(range(len(ep_vals))):
                if t + 1 < len(ep_rewards):
                    delta = ep_rewards[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rewards[t] - ep_vals[t]

                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                ep_advantages.insert(0, advantage)

            advantages.extend(ep_advantages)
        
        advantages = torch.tensor(advantages, dtype=torch.float)
        returns = advantages + self.critic(obs).squeeze().detach()
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        return normalized_advantages, returns

    def restore_savestate(self, checkpoint):
        model = ActorCritic(self.obs_dim, self.act_dim)
        model.load_state_dict(torch.load(checkpoint))
        self.policy = model

    def validate(self, max_iter, env = LunarContinuous().make_environment()):
        self.policy.eval()
        val_rews = []
        val_dur = []
        iter = 0
        for _ in range(0, max_iter) :
                obs, _ = env.reset()
                done = False

                t = 0
                ep_ret = 0

                while not done:
                    t += 1
                    action = self.actor(obs)
                    obs, rew, terminated, truncated, _ = env.step(action.detach().numpy())
                    done = terminated | truncated

                    ep_ret += rew
                    
                val_rews.append(ep_ret)
                val_dur.append(t)
        return val_rews,  val_dur

    def test(self, env = LunarContinuous(True).make_environment()):
        self.policy.test()
        while True:
                obs, _ = env.reset()
                done = False
                while not done:
                    action = self.actor(obs)
                    obs, _, terminated, truncated, _ = env.step(action.detach().numpy())
                    done = terminated | truncated

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
        self.lam = 0.98                                 # Lambda Parameter for GAE 
        self.clip = 0.2                                 # Using the recommended value of 0.2, helps define the threshold to clip the ratio during SGA
        self.max_grad_norm = 0.5                        # Gradient Clipping threshold
        self.lr_gamma = 0.9998

        # Miscellaneous parameters
        self.save_freq = 10                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results
        self.num_workers = 8                            # Sets the ammount of workers to parallelise rollouts

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
        avg_rollout_t = str(round(self.logger['avg_rollout_t']/ 1e9, 2))

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

        # Round decimal places for prettier print
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
        print(f"A parallel rollout took on average: {avg_rollout_t} secs", flush=True)
        print(f"Current learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)