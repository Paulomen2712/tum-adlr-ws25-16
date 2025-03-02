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
    """PPO Algorithm Implementation. Only trains base policy, which can include Object Prop Encoder"""

    def __init__(self, summary_writter=None, env = LunarContinuous, policy_class = ActorCritic, activation=nn.ReLU, env_args={}, **hyperparameters):
        """
			Initializes the PPO model, including hyperparameters.
		"""
        # Extract environment information
        self.env_args = env_args
        self.env_class = env
        self.env = env(**env_args)
        self.obs_dim, self.act_dim =  self.env.get_environment_shape()
         
        # Initialize hyperparameters for training with PPO
        self.summary_writter = summary_writter
        self._init_hyperparameters(hyperparameters)
        self.storage = Storage(self.num_steps, self.num_envs, self.obs_dim, self.act_dim, self.gamma, self.lam, self.normalize_advantages, self.device )

        # Initialize actor and critic
        self.policy = policy_class(self.obs_dim, self.act_dim, lr=self.lr, gamma=self.lr_gamma, hidden_dims=self.hidden_dims, activation=activation)
        self.actor = self.policy.actor                                              
        self.critic = self.policy.critic

        # Initialize optimizers for actor and critic
        self.actor_optim = self.policy.actor_optim  
        self.critic_optim = self.policy.critic_optim
        
        self.policy.to(self.device)

        self.logger = {
			'delta_t': time.time_ns(),
			'i_so_far': 0,          # iterations simulated so far
			'batch_rews': 0,        # episodic returns in current batch
            'rollout_compute': 0,   # compute required for a rollout
            'grad_compute': 0,      # compute required for the total gradient update process
			'actor_losses': [],     # losses of actor network in current batch
            'val_rew': None,        # current training time validation rewards
            'val_dur': None,        # current training time validation duration
            'kls': [],              # kl divergence of actor network in current batch
            'lr': self.lr,          # current learning rate
		}

    def train(self):
        """
            Train the base policy.
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
                    batch_values, pred_batch_log_probs, _ = self.policy.evaluate(batch_obs, batch_acts)
                    
                    actor_loss, kl = self.update_actor(pred_batch_log_probs, batch_log_probs, batch_advantages)
                    self.update_critic(batch_values, batch_returns)

                    loss.append(actor_loss.detach().item())
                    kls.append(kl.item())

                self.logger['actor_losses'].append(np.mean(loss))
                self.logger['kls'].append(np.mean(kls))
            # self.logger['lr'] = self.actor_scheduler.get_last_lr()[0]

            grad_end = time.time_ns()
            self.logger['grad_compute'] = (grad_end - grad_start) / 1e9

            if self.val_freq > 0 and (it+1) % self.val_freq == 0:
                with torch.no_grad():
                    val_rew, val_dur = self.validate(self.val_iter)
                self.logger['val_rew'] = np.mean(val_rew)
                self.logger['val_dur'] = np.mean(val_dur)

            self._log_summary()

            if self.save_freq > 0 and (it+1) % self.save_freq == 0:
                save_path = self._get_save_path(it+1)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.policy.state_dict(), save_path)
                
    def rollout(self):
        """
            Collects batch of simulated data.
        """
        self.storage.clear()

        next_obs, next_done = self.env.reset()
        for _ in range(self.num_steps):
            obs = next_obs.copy() 

            actions, logprobs, values = self.policy.act(torch.from_numpy(next_obs).to(self.device, torch.float32))
            next_obs, rewards, next_done = self.env.step(actions.cpu().numpy())

            self.storage.store_batch(obs, actions, logprobs, rewards, values, next_done)
        self.storage.compute_advantages(self.policy.get_value(torch.from_numpy(next_obs).to(self.device, torch.float32)))
        return self.storage.get_rollot_data()

    def update_actor(self, pred_log_probs, log_probs, advantages):
        """Performs ppo gradient step of the actor module. Also returns an estimate for the current KL divergence"""
        log_ratios = pred_log_probs - log_probs
        ratios = torch.exp(log_ratios)
        surr1 = -ratios * advantages
        surr2 = -torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
        with torch.no_grad():
            kl = ((ratios - 1) - log_ratios).mean()
        actor_loss = (torch.max(surr1, surr2)).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.policy.actor_logstd, self.max_grad_norm)
        self.actor_optim.step()
        return actor_loss.detach(), kl

    def update_critic(self,  values, returns):
        """Performs gradient step of the critic module"""
        critic_loss = nn.MSELoss()(values, returns)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()

    def restore_savestate(self, checkpoint):
        """Loads policy from savestate"""
        model = ActorCritic(self.obs_dim, self.act_dim)
        model.load_state_dict(torch.load(checkpoint))
        self.policy = model

    def validate(self, val_iter, should_record=False):
        """Runs for a set ammount the policy simulations and returns mean rewards and mean timesteps. If should_record=True stores videos locally"""
        if should_record:
            env = self.env_class(num_envs=val_iter,should_record='True',**self.env_args)
        else:
            env = self.env_class(num_envs=val_iter,**self.env_args)
        obs, done  = env.reset()
        self.policy.to(self.device)
        t = np.array([0]*val_iter, dtype=float)
        ep_ret = np.array([0]*val_iter, dtype=float)
        t_sim = 0

        while not all(done) and t_sim <= self.num_steps:
            t_sim+=1
            not_done = np.array([1]*val_iter, dtype=float) - done
            t += not_done
            action, _ = self.policy.sample_action(torch.Tensor(obs).to(self.device))
            obs, rew, next_done = env.step(action.cpu().numpy())
            ep_ret += rew * not_done
            done |= next_done
        env.close()   
        return np.mean(ep_ret),  np.mean(t)

    def test(self):
        """Runs endlessly the policy simulations. These runs are rendered"""
        self.policy.cpu()
        env = self.env_class(num_envs=1,render_mode='human',**self.env_args)
        while True:
                obs, done = env.reset()
                t = 1
                while not done[0]:
                    action, _ = self.policy.sample_action(torch.Tensor(obs))
                    obs, _, done = env.step(action.numpy())
                    t+=1

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters
        """
        #idea log hyperparameters from yaml config, but they can still be overriden by specifying on creation
        config_hyperparameters = self.env.load_hyperparameters()
        for param, val in config_hyperparameters.items():
            setattr(self, param, val)

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        self.base_train_it = self.total_base_train_steps // (self.num_envs * self.num_steps)

    def _get_save_path(self, iteration):
        """Returns the path to store the current checkpoint"""
        if self.summary_writter is None:
            return f'./ppo_checkpoints/non_wandb/ppo_policy_{iteration}.pth'
        else:
            return f'./ppo_checkpoints/{wandb.run.name}/ppo_policy_{iteration}.pth'

    def _log_summary(self):
        """Prints summary of the current base policy training epoch to std out and logs to wandb"""
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
                "average_episode_rewards": avg_ep_rews,
                "average_loss": avg_actor_loss,
                "learning_rate": lr,
                "iteration_compute": delta_t
            })
        delta_t = str(round(delta_t, 2))

        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        print(flush=True)
        print(f"-------------------- Iteration {i_so_far}/{self.base_train_it} --------------------", flush=True)
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