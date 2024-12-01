import os
from dataclasses import asdict
import wandb

class SummaryWritter():
    pass

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

    def save_model(self, model_path):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))