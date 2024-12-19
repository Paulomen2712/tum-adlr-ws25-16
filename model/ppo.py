import torch
from networks.actorcritic import ActorCritic
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
        self.env_class = env
        self.env = env()
        self.obs_dim, self.act_dim =  self.env.get_environment_shape()
         
        # Initialize hyperparameters for training with PPO
        self.summary_writter = summary_writter
        self._init_hyperparameters(hyperparameters)
        self.device = 'cuda'
        self.storage = Storage(self.num_steps, self.num_envs, self.obs_dim, self.act_dim, self.gamma, self.lam, self.device )

        # Initialize actor and critic
        self.policy = policy_class(self.obs_dim, self.act_dim, hidden_dims=self.policy_hidden_dims, lr=self.lr, gamma=self.lr_gamma)
        self.actor = self.policy.actor                                              
        self.critic = self.policy.critic
        self.base_encoder = None
        self.adapt_encoder = None
        

        # Initialize optimizers for actor and critic
        self.actor_optim = self.policy.actor_optim  
        self.critic_optim = self.policy.critic_optim
        self.adapt_optim=None

        self.actor_scheduler = self.policy.actor_scheduler
        self.critic_scheduler = self.policy.actor_optim
        
        self.policy.to(self.device)

        self.logger = {
			'delta_t': time.time_ns(),
			'i_so_far': 0,          # iterations simulated so far
			'batch_rews': 0,       # episodic returns in current batch
            'rollout_compute': 0,
            'grad_compute': 0,
			'actor_losses': [],     # losses of actor network in current batch
            'val_rew': None,
            'val_dur': None,
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
            if self.anneal_lr:
                frac = it / (self.base_train_it + self.anneal_discount)
                new_lr = self.lr * (1.0 - frac)

                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                # Log learning rate
                self.lr = new_lr
                self.logger['lr'] = self.lr

            rollout_start = time.time_ns()  
            obs, acts, log_probs, advantages, returns = self.rollout()
            rollout_end = time.time_ns() 

            self.logger['i_so_far'] = it + 1
            self.logger['batch_rews'] = self.storage.get_average_episode_rewards()
            self.logger['rollout_compute'] = (rollout_end - rollout_start) / 1e9


            batch_size = obs.size(0)
            inds = np.arange(batch_size)
            sgdbatch_size = batch_size // self.n_sgd_batches
            loss = []
            kls = []

            grad_start = time.time_ns() 
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
                    batch_advantages = advantages[idx]
                    batch_returns = returns[idx]
                    batch_values, pred_batch_log_probs = self.policy.evaluate(batch_obs, batch_acts)
                    
                    actor_loss, kl = self.update_actor(pred_batch_log_probs, batch_log_probs, batch_advantages)
                    self.update_critic(batch_values, batch_returns)

                    loss.append(actor_loss.detach().item())
                    kls.append(kl.item())

                self.logger['actor_losses'].append(np.mean(loss))
                self.logger['kls'].append(np.mean(kls))

            grad_end = time.time_ns()
            self.logger['grad_compute'] = (grad_end - grad_start) / 1e9

            if self.val_freq > 0 and (it+1) % self.val_freq == 0:
                with torch.no_grad():
                    val_rew, val_dur = self.validate(self.val_iter)
                self.logger['val_rew'] = np.mean(val_rew)
                self.logger['val_dur'] = np.mean(val_dur)

            self._log_summary()

			# Save model every couple iterations
            if self.save_freq > 0 and (it+1) % self.save_freq == 0:
                save_path = self._get_save_path(it+1)
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
        self.storage.clear()

        next_obs, _ = self.env.reset()
        for _ in range(self.num_steps):
            obs = next_obs.copy() 

            actions, logprobs, values = self.policy.act(torch.from_numpy(next_obs).to(self.device))
            next_obs, rewards, dones = self.env.step(actions.cpu().numpy())

            self.storage.store_batch(obs, actions, logprobs, rewards, values, dones)
        self.storage.compute_advantages(self.policy.get_value(torch.from_numpy(next_obs).to(self.device)))
        return self.storage.get_rollot_data()

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

    def validate(self, val_iter, should_record=False):
        if should_record:
            env = self.env_class(num_envs=val_iter,should_record='True')
        else:
            env = self.env_class(num_envs=val_iter)
        policy = self.policy
        obs, done  = env.reset()

        t = np.array([0]*val_iter, dtype=float)
        ep_ret = np.array([0]*val_iter, dtype=float)
        t_sim = 0
        while not all(done) and t_sim <= self.num_steps:
            t_sim+=1
            not_done = np.array([1]*val_iter, dtype=float) - done
            t += not_done
            action = policy.sample_action(torch.Tensor(obs).to(self.device))
            obs, rew, next_done = env.step(action.cpu().numpy())
            done |= next_done
            ep_ret += rew * not_done
            
        env.close()
        return np.mean(ep_ret),  np.mean(t)
    
    def test(self):
        self.policy.cpu()
        env = self.env_class(num_envs=1,render_mode='human')
        while True:
                obs, done = env.reset()
                print(obs, flush=True)
                while not done[0]:
                    action = self.policy.sample_action(torch.Tensor(obs))
                    obs, _, done = env.step(action.numpy())

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters
        """

        config_hyperparameters = self.env.load_hyperparameters()
        for param, val in config_hyperparameters.items():
            setattr(self, param, val)

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        self.base_train_it = self.total_timesteps // (self.num_envs*self.num_steps)

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
        rollout_t = str(round(self.logger['rollout_compute'], 2))
        grad_t = str(round(self.logger['grad_compute'], 2))

        i_so_far = self.logger['i_so_far']
        avg_ep_rews = self.logger['batch_rews'].item()
        avg_actor_loss = np.mean([losses for losses in self.logger['actor_losses']])
        avg_kl = np.mean([kl for kl in self.logger['kls']])

        #log to wandb
        if self.summary_writter is not None:
            self.summary_writter.save_dict({
                "simulated_iterations": i_so_far,
                "simulated_timesteps": i_so_far*self.num_envs*self.num_steps,
                "average_episode_rewards": avg_ep_rews,
                "average_loss": avg_actor_loss,
                "learning_rate": lr,
                "iteration_compute": delta_t
            })
        delta_t = str(round(delta_t, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Simulated timesteps: {i_so_far*self.num_envs*self.num_steps}")
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Average KL Divergence: {avg_kl}", flush=True)
        print(f"Iteration took: {delta_t} secs, of which rollout took {rollout_t} secs and gradient updates took {grad_t} secs", flush=True)
        print(f"Current learning rate: {lr}", flush=True)

        if(self.logger['val_rew'] is not None):
            avg_val_rews = str(round(self.logger['val_rew'], 2))
            val_durs = self.logger['val_dur']
            print(f"Average Validation Return: {avg_val_rews}", flush=True)
            print(f"Average Validation Duration: {val_durs} secs", flush=True)

            if self.summary_writter is not None:
                self.summary_writter.save_dict({
                    "val_rews": self.logger['val_rew'],
                    "val_durs": self.logger['val_dur']
                })

            self.logger['val_dur'] = None
            self.logger['val_rew'] = None


        print(f"------------------------------------------------------", flush=True)
        print(flush=True)