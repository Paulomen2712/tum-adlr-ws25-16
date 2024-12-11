import torch
from model.network import ActorCritic
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
			'batch_rews': 0,       # episodic returns in current batch
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

        for t in range(0, 10000):   
            obs, acts, log_probs, advantages = self.rollout()
            iteration += 1

            self.logger['t_so_far'] = t_sim
            self.logger['i_so_far'] = iteration

            returns = advantages + self.policy.get_value(obs)
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
        self.num_envs = 1
        self.num_steps= 1200
        self.storage_size = 20

        device = 'cpu'

        obs = torch.zeros((self.storage_size, self.num_steps, self.num_envs, self.obs_dim))#.to(device)
        actions = torch.zeros((self.storage_size,self.num_steps, self.num_envs, self.act_dim))#.to(device)
        logprobs = torch.zeros((self.storage_size,self.num_steps, self.num_envs, 1))#.to(device)
        rewards = torch.zeros((self.storage_size,self.num_steps, self.num_envs, 1))#.to(device)
        dones = torch.zeros((self.storage_size,self.num_steps, self.num_envs, 1))#.to(device)
        values = torch.zeros((self.storage_size, self.num_steps, self.num_envs, 1))#.to(device)

        global_step = 0
        start_time = time.time()
        # next_obs = torch.Tensor(self.env.reset()).to(device)
        # next_done = torch.zeros(self.num_envs).to(device)
        next_obs, next_done = self.env.reset()
        next_obs, next_done = torch.Tensor(next_obs), torch.Tensor([float(next_done)])#.to(device)
        for i in range(0,self.storage_size):

            # ep_rews = []
            # ep_vals = []
            # ep_dones = []

            # obs, done = self.env.reset()

            for step in range(self.num_steps):
                obs[i,step] = next_obs
                dones[i,step] = next_done

                action, logprob, val = self.policy.act(next_obs)
                next_obs, rew, next_done = self.env.step(action)


                values[i,step] = val.flatten()
                actions[i,step] = torch.Tensor(action)#.to(device)
                logprobs[i,step] = logprob
                rewards[i,step] = torch.tensor(rew).view(-1)
                next_obs, next_done = torch.Tensor(next_obs), torch.Tensor([float(next_done)])#.to(device)
            
            # batch_advantages.extend(self.gae(ep_rews, ep_vals, ep_dones))

        

        # batch_advantages = torch.tensor(batch_advantages, dtype=torch.float)
        # batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        # batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        # batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)

        self.logger['batch_rews'] = torch.mean(rewards)
        # self.logger['batch_lens'] = batch_lens

        return self.format_tensor(obs), self.format_tensor(actions), self.format_tensor(logprobs), self.format_tensor(self.gae(rewards, values, dones))
    
    def format_tensor(self, tensor):
        return tensor.transpose(1, 2).flatten(start_dim=0, end_dim=1)

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

    def gae(self, rewards, values, dones):
        """
            Computes generalized advantage estimation (see https://arxiv.org/abs/1506.02438 page 4)
        """
        device='cpu'
        advantages = torch.zeros_like(rewards)#.to(device)

        # Add one additional step value for bootstrap
        values_next = torch.zeros_like(values)#.to(device)
        values_next[:, :-1] = values[:, 1:]  # Next values shifted for GAE
        values_next[:, -1] = values[:, -1]  # Bootstrap with last value

        # Compute deltas (TD residuals)
        deltas = rewards + self.gamma * (1 - dones) * values_next - values

        # Compute GAE advantage in reverse
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:  # Last timestep
                advantages[:, t] = deltas[:, t]
            else:
                advantages[:, t] = deltas[:, t] + self.gamma * self.lam * (1 - dones[:, t]) * advantages[:, t + 1]

        # Flatten the first two dimensions for returning GAE across all steps/envs
        return self.format_tensor(advantages).unsqueeze(-1)

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
        self.policy.eval()
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

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = self.logger['batch_rews'].item()
        #avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

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
        #avg_actor_loss = str(round(avg_actor_loss, 5))

        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        #print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Current learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)