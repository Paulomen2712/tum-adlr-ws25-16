import os
from dataclasses import asdict
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
import io

class SummaryWritter():
    def log(self, cur_iteration):
        pass
        # avg_ep_lens = str(round(avg_ep_lens, 2))
        # avg_ep_rews = str(round(avg_ep_rews, 2))
        # avg_actor_loss = str(round(avg_actor_loss, 5))

        # print(flush=True)
        # print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        # print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        # print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        # print(f"Average Loss: {avg_actor_loss}", flush=True)
        # print(f"Timesteps So Far: {t_so_far}", flush=True)
        # print(f"Iteration took: {delta_t} secs", flush=True)
        # print(f"Current learning rate: {lr}", flush=True)
        # print(f"------------------------------------------------------", flush=True)
        # print(flush=True)

class WandbSummaryWritter():
    """Logger for wandb.com"""

    def __init__(self, project, config):
        wandb.login()

        wandb.init(project=project, config=config)

        wandb.run.name = project + '-' + wandb.run.name.split("-")[-1]


    def save_dict(self, dict):
        wandb.log(dict)

    def save_video(self, path, name="environment_run_recording"):
        wandb.log({name: wandb.Video(path, fps=4, format="mp4")})

    def stop(self):
        wandb.finish()

    def save_model(self, model, model_name):
        torch.save(model.state_dict(), f"{wandb.run.dir}/{model_name}.pth")
        artifact = wandb.Artifact(model_name, type="model")
        artifact.add_file(f"{wandb.run.dir}/{model_name}.pth", f"{model_name}.pth")
        wandb.log_artifact(artifact)

    def save_histogram(self, data, data_name = 'Data', num_bins = 10):
        median_value = np.median(data)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(data, bins=num_bins, edgecolor='black', alpha=0.7, label=data_name)

        ax.axvline(median_value, color='red', linestyle='dashed', linewidth=2, label=f'Median = {median_value}')

        ax.set_xlabel('Value Range')
        ax.set_ylabel('Frequency')
        ax.legend()
        wandb.log({data_name: wandb.Image(fig)})

        plt.close(fig)

    def save_file(self, path):
        wandb.save(path, base_path=os.path.dirname(path))