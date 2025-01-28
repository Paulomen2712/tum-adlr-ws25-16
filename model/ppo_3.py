import torch
from networks.base_policy import ActorCriticWithEncoder
from networks.adaptive_module import AdaptiveActorCritic
# from networks.adaptive_module_2 import AdaptiveActorCritic
from env.wrappers import LunarContinuous
import torch.nn as nn
import numpy as np
import time
import wandb
import os
from utils.storage import Storage, AdaptStorage


class PPO:
    """PPO Algorithm Implementation."""

    def __init__(self, summary_writter=None, env = LunarContinuous, policy_class = ActorCriticWithEncoder, **hyperparameters):
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
        self.storage = Storage(self.num_steps, self.num_envs, self.obs_dim, self.act_dim, self.gamma, self.lam, self.normalize_advantages, self.device)

        self.adp_env = env(num_envs=self.num_adp_envs)
        self.adp_storage = AdaptStorage(self.adp_num_steps, self.num_adp_envs, self.obs_dim, self.device )

        # Initialize actor and critic
        self.policy = policy_class(self.obs_dim, self.act_dim, lr=self.lr, hidden_dims=self.hidden_dims, encoder_hidden_dims=self.encoder_hidden_dims)
        self.actor = self.policy.actor                                              
        self.critic = self.policy.critic
        self.adapt_policy = AdaptiveActorCritic(self.obs_dim, self.act_dim, lr=self.adp_lr, encoder_hidden_dims=self.adp_encoder_hidden_dims, history_len=self.history_len)
        self.adpt_module = self.adapt_policy.encoder
        

        # Initialize optimizers for actor and critic
        self.actor_optim = self.policy.actor_optim  
        self.critic_optim = self.policy.critic_optim
        self.adapt_optim=self.adapt_policy.optim
        
        self.policy.to(self.device)

        self.logger = {
			'delta_t': time.time_ns(),
			'i_so_far': 0,          # iterations simulated so far
			'batch_rews': 0,       # episodic returns in current batch
            'rollout_compute': 0,
            'grad_compute': 0,
			'actor_losses': [],     # losses of actor network in current batch
            'critic_losses': [],
            'adp_losses': [],
            'val_rew': None,
            'val_dur': None,
            'kls': [],
            'lr': self.lr,           
            'adp_lr': self.adp_lr,
            'max_val_rew': 0
		}

    def train(self):
        """
            Train the actor/critic network.
        """
        self.train_base()
        self.train_adaptive_module()

    def train_base(self):
        self.policy.train()
        self.logger['delta_t'] = time.time_ns()

        for it in range(0, self.base_train_it): 
            if self.anneal_lr:
                frac = it / (self.base_train_it + self.anneal_discount)
                lr = self.actor_lr * (1.0 - frac)

                self.actor_optim.param_groups[0]["lr"] = lr
                self.critic_optim.param_groups[0]["lr"] = lr
                # Log learning rate
                self.lr = lr
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
            act_loss = []
            crit_loss = []
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
                    batch_values, pred_batch_log_probs, pred_batch_entropies, true_z, encoded_z = self.policy.evaluate(batch_obs, batch_acts)
                    
                    actor_loss, kl = self.update_actor(pred_batch_log_probs, batch_log_probs, batch_advantages, pred_batch_entropies)
                    # actor_loss, kl = self.update_actor_v2(pred_batch_log_probs, batch_log_probs, batch_advantages, pred_batch_entropies, true_z, encoded_z)
                    critic_loss = self.update_critic(batch_values, batch_returns)

                    act_loss.append(actor_loss.detach().item())
                    crit_loss.append(critic_loss.detach().item())
                    kls.append(kl.item())

                self.logger['actor_losses'].append(np.mean(act_loss))
                self.logger['critic_losses'].append(np.mean(crit_loss))
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

			# Save model every couple iterations
            if self.save_freq > 0 and (it+1) % self.save_freq == 0:
                save_path = self._get_save_path(it+1)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.policy.state_dict(), save_path)

    def train_adaptive_module(self):
        self.adapt_policy.set_policy(self.policy)
        self.adapt_policy.to(self.device)
        for ad_it in range(0, self.adp_train_it):
            if self.anneal_lr:
                frac = ad_it / (self.adp_train_it + self.anneal_discount) 
                adp_lr = self.adp_lr * (1.0 - frac)

                self.adapt_optim.param_groups[0]["lr"] = adp_lr
                # Log learning rate
                self.adp_lr = adp_lr
                self.logger['adp_lr'] = self.adp_lr = adp_lr

            rollout_start = time.time_ns() 
            self.adapt_policy.clear_history()
            obs, values = self.adpt_rollout()
            rollout_end = time.time_ns() 

            self.logger['i_so_far'] = ad_it + 1
            self.logger['rollout_compute'] = (rollout_end - rollout_start) / 1e9


            batch_size = obs.size(0)
            inds = np.arange(batch_size)
            sgdbatch_size = batch_size // self.n_sgd_batches
            loss = []

            grad_start = time.time_ns() 
            for _ in range(self.n_updates_per_iteration): 

                #SGD
                np.random.shuffle(inds)
                for start in range(0, batch_size, sgdbatch_size):
                    end = start + sgdbatch_size
                    idx = inds[start:end]
                    
                    #Restrict values to current batch
                    batch_obs = obs[idx]
                    batch_values = values[idx]
                    pred_batch_values = self.adapt_policy.evaluate(batch_obs)
                    
                    # BUG: Using batch_values is wrong of the critic is wrong. Should use base_encoder Z
                    adp_loss = self.update_adpt(pred_batch_values, batch_values)
                    loss.append(adp_loss.detach().item())
                self.logger['adp_losses'].append(np.mean(loss))

            grad_end = time.time_ns()
            self.logger['grad_compute'] = (grad_end - grad_start) / 1e9
            self._log_summary_adp()

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
            obs = next_obs.copy() 

            actions, logprobs, values = self.policy.act(torch.from_numpy(next_obs).to(self.device, torch.float32))
            next_obs, rewards, next_done = self.env.step(actions.cpu().numpy())

            self.storage.store_batch(obs, actions, logprobs, rewards, values, next_done)
        self.storage.compute_advantages(self.policy.get_value(torch.from_numpy(next_obs).to(self.device, torch.float32)))
        return self.storage.get_rollot_data()

    def adpt_rollout(self):
        self.adp_storage.clear()

        next_obs, _ = self.adp_env.reset()
        for _ in range(self.adp_num_steps):
            obs = next_obs.copy() 

            actions= self.policy.sample_action(torch.from_numpy(next_obs).to(self.device, torch.float32))
            z = self.policy.encode(torch.from_numpy(next_obs).to(self.device)).detach()[:, -1]

            next_obs, _, _ = self.adp_env.step(actions.cpu().numpy())

            self.adp_storage.store_obs(obs, z)
        return self.adp_storage.get_rollot_data()

    def update_actor(self, pred_log_probs, log_probs, advantages, entropies):
        log_ratios = pred_log_probs - log_probs
        ratios = torch.exp(log_ratios)
        surr1 = -ratios * advantages
        surr2 = -torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
        with torch.no_grad():
            kl = ((ratios - 1) - log_ratios).mean()
        actor_loss = (torch.max(surr1, surr2)).mean()
        actor_loss -= self.entropy_coef * entropies.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.policy.actor_logstd, self.max_grad_norm)
        self.actor_optim.step()
        return actor_loss.detach(), kl

    def update_actor_v2(self, pred_log_probs, log_probs, advantages, entropies, true_z, encoded_z):
        encoder_loss = nn.MSELoss()(true_z.squeeze(), encoded_z.squeeze())
        encoder_loss /= 10

        log_ratios = pred_log_probs - log_probs
        ratios = torch.exp(log_ratios)
        surr1 = -ratios * advantages
        surr2 = -torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages

        with torch.no_grad():
            kl = ((ratios - 1) - log_ratios).mean()

        actor_loss = (torch.max(surr1, surr2)).mean()
        actor_loss -= self.entropy_coef * entropies.mean()

        actor_loss += encoder_loss

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.policy.actor_logstd, self.max_grad_norm)
        self.actor_optim.step()
        return actor_loss.detach(), kl

    def update_critic(self,  values, returns):
        critic_loss = nn.MSELoss()(values.squeeze(), returns.squeeze())
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()
        return critic_loss

    def update_adpt(self,  pred_values, values):
        pred_size = pred_values.shape[0]
        values_size = values.shape[0]

        if pred_size > values_size:
            padding_size = pred_size - values_size
            padding = torch.zeros(padding_size, device=values.device)  # Create padding with zeros
            values = torch.cat([values, padding], dim=0)  # Concatenate the padding along the first dimension

        adapt_loss = nn.MSELoss()(pred_values.squeeze(), values.squeeze())
        
        self.adapt_optim.zero_grad()
        adapt_loss.backward()
        # if self.clip_grad:
        #     nn.utils.clip_grad_norm_(self.adpt_module.parameters(), self.max_grad_norm)
        self.adapt_optim.step()
        return adapt_loss

    def restore_savestate(self, base_checkpoint, adp_checkpoint):
        base_model = ActorCriticWithEncoder(self.obs_dim, self.act_dim)
        base_model.load_state_dict(torch.load(base_checkpoint))
        self.policy = base_model

        adp_model = AdaptiveActorCritic(self.obs_dim, self.act_dim, lr=self.adp_lr)
        adp_model.load_state_dict(torch.load(adp_checkpoint))
        self.adapt_policy = adp_model
        self.adapt_policy.set_policy(base_model)

    def validate(self, val_iter, should_record=False, use_adaptive=False, interupt=True):
        if should_record:
            env = self.env_class(num_envs=val_iter,should_record='True')
        else:
            env = self.env_class(num_envs=val_iter)
        if use_adaptive:
            self.adapt_policy.clear_history()
            policy = self.adapt_policy
        else:
            policy = self.policy
        obs, done  = env.reset()

        t = np.array([0]*val_iter, dtype=float)
        ep_ret = np.array([0]*val_iter, dtype=float)
        t_sim = 0

        max_steps = self.num_steps if interupt else 2500

        while not all(done) and t_sim <= max_steps:
            t_sim+=1
            not_done = np.array([1]*val_iter, dtype=float) - done
            t += not_done
            action = policy.sample_action(torch.Tensor(obs).to(self.device))
            obs, rew, next_done = env.step(action.cpu().numpy())
            done |= next_done
            ep_ret += rew * not_done
        
        env.close()

        if self.logger['val_rew'] is None:
            self.logger['val_rew'] = ep_ret
            
        return ep_ret, t

    def validate_encoders_single_rollout(self):
        self.adapt_policy.clear_history()

        true_wind_vals = []
        base_z = []
        adpt_z = []

        env = self.env_class(num_envs=1,should_record='True')
        obs, done = env.reset()

        while not done[0]:
            true_wind_vals.append(obs[0, -1])

            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            action = self.adapt_policy.sample_action(obs_tensor)

            base_output = self.policy.encoder(obs_tensor).detach().cpu().numpy().flatten()[0]
            adpt_output = self.adapt_policy.encode(obs_tensor).detach().cpu().numpy().squeeze()[-1]

            base_z.append(base_output)
            adpt_z.append(adpt_output)
            obs, _, done = env.step(action.cpu().numpy())
            
        return true_wind_vals, base_z, adpt_z

    def validate_encoders(self, num_envs = 100, num_steps = 20):
        self.adapt_policy.clear_history()
        adpt_policy = self.adapt_policy

        env = self.env_class(num_envs=num_envs)
        
        obs, done  = env.reset()

        for _ in range(num_steps):
            adpt_policy_action = adpt_policy.sample_action(torch.Tensor(obs).to(self.device))

            obs, _, _ = env.step(adpt_policy_action.cpu().numpy())

        
        true_winds = obs[:, -1]
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        base_output = self.policy.encoder(obs_tensor).cpu().detach().numpy().flatten()
        adpt_output = self.adapt_policy.encode(obs_tensor).cpu().detach().numpy().squeeze()[:, -1]

        return true_winds, base_output, adpt_output

    def test(self, use_adaptive=True):
        if use_adaptive:
            policy = self.adapt_policy
            self.adapt_policy.clear_history()
        else:
            policy = self.policy
        policy.cpu()
        env = self.env_class(num_envs=1,render_mode='human')
        while True:
                obs, done = env.reset()
                while not done[0]:
                    action = policy.sample_action(torch.Tensor(obs))
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
        self.base_train_it = self.total_base_train_steps // (self.num_envs * self.num_steps)
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
        avg_critic_loss = np.mean([losses for losses in self.logger['critic_losses']])
        avg_kl = np.mean([kl for kl in self.logger['kls']])



        #log to wandb
        if self.summary_writter is not None:
            self.summary_writter.save_dict({
                "simulated_iterations": i_so_far,
                "average_episode_rewards": avg_ep_rews,
                "average_actor_loss": avg_actor_loss,
                "average_critic_loss": avg_critic_loss,
                "learning_rate": lr,
                "iteration_compute": delta_t
            })
        
        delta_t = str(round(delta_t, 2))

        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        print(flush=True)
        print(f"-------------------- Iteration {i_so_far}/{self.base_train_it} --------------------", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
        print(f"Average Critic Loss: {avg_critic_loss}", flush=True)
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
                    "val_durs": self.logger['val_dur'],
                })

            self.logger['val_dur'] = None
            self.logger['val_rew'] = None


        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

    def _log_summary_adp(self):
        adp_lr = self.logger['adp_lr']
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        rollout_t = str(round(self.logger['rollout_compute'], 2))
        grad_t = str(round(self.logger['grad_compute'], 2))

        i_so_far = self.logger['i_so_far']
        avg_loss = np.mean([losses for losses in self.logger['adp_losses']])

        #log to wandb
        if self.summary_writter is not None:
            self.summary_writter.save_dict({
                "simulated_iterations": i_so_far,
                "average_adapt_loss": avg_loss,
                "adp_learning_rate": adp_lr,
                "iteration_compute": delta_t
            })
        delta_t = str(round(delta_t, 2))

        avg_loss = str(round(avg_loss, 5))

        print(flush=True)
        print(f"-------------------- Iteration{i_so_far}/{self.adp_train_it} --------------------", flush=True)
        print(f"Average adp Loss: {avg_loss}", flush=True)
        print(f"Iteration took: {delta_t} secs, of which rollout took {rollout_t} secs and gradient updates took {grad_t} secs", flush=True)
        print(f"Current adp learning rate: {adp_lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)