import gym
from gym.vector import SyncVectorEnv
from gym.wrappers import RecordVideo
from gym.envs.box2d import LunarLander
from env.lunar_lander import LunarLanderWithWind
import numpy as np
import yaml

class VectorEnvironment(SyncVectorEnv):
    """ Base class for wrapping environments. """
    def __init__(self, env_fns):
        """
            Initialize parameters.
        """
        super().__init__(env_fns)

    def _make_environment(self, **args):
        """
            Makes a new environment.
        """
        pass  # Placeholder for environment-specific implementation

    def _get_config_path(self):
        """
            Returns the hyperparameter yaml file path.
        """
        pass  # Placeholder for environment-specific implementation
    
    def get_environment_shape(self):
        """
            Returns tuple (observatio_space_shape, action_space_shape).
        """
        pass  # Placeholder for environment-specific implementation
    
    def make_environment_for_recording(self, env_id, episode_trigger=lambda _: True, **args):
        """
            Additionally wraps the environment for recording.
        """
        return RecordVideo(self._make_environment(render_mode = 'rgb_array', **args), video_folder="videos", name_prefix=f'rl-video{env_id}', episode_trigger=episode_trigger)

    def load_hyperparameters(self):
        with open(self._get_config_path(), 'r') as f:
            return yaml.safe_load(f)

class LunarContinuous(VectorEnvironment):
    """ OpenAi Lunar Continuous Environment Wrapper. """

    def __init__(self,  **args):
        """
            Initialize parameters.

            Parameters:
                render_mode (string): mode to render the environment
        """
        config = self.load_hyperparameters()
        if 'num_envs' in args:
            num_envs = args.get('num_envs')
            del args['num_envs']
        else:
            num_envs = config.get('num_envs', 1)
            
        if 'should_record' in args and args.get('should_record'):
            del args['should_record']
            super().__init__( [lambda i=i: self.make_environment_for_recording(i,**args) for i in range(num_envs)])
        else:
            super().__init__( [lambda: self._make_environment( **args) for _ in range(num_envs)])

    def _make_environment(self, **args):
        env = LunarLander(continuous=True, **args)
        return env
    
    def _get_config_path(self):
        return "./configs/LunarLander.yaml"
    
    def get_environment_shape(self):
        return 8, 2
    
    def reset(self):
        obs, _ = super().reset()
        done = np.array([False]*self.num_envs) # done is always false after reset
        return obs, done
    
    def step(self, action):
        obs, reward, terminated, truncated, _ = super().step(action)
        done = terminated | truncated
        return obs, reward, done
    
    
class LunarLanderWithKnownWind(LunarContinuous):
    """
        OpenAi Lunar Continuous Environment Wrapper that adds the wind to the observation space.
    """
    def __init__(self, **args):
        config = self.load_hyperparameters()
        min_wind_power = config.get('min_wind_power', 15)
        max_wind_power = config.get('max_wind_power', 50)
        super().__init__( **args, min_wind_power=min_wind_power, max_wind_power=max_wind_power)

    def _make_environment(self, **args):
        return LunarLanderWithWind(**args)
    
    def get_environment_shape(self):
        return 9, 2
    
class LunarLanderWithUnknownWind(LunarLanderWithKnownWind):
    """ OpenAi Lunar Continuous Environment Wrapper with wind enabled environment wrapper. """
    def __init__(self, **args):
        super().__init__(**args)

    def get_environment_shape(self):
        return 8, 2
    
    def reset(self):
        obs, done = super().reset()
        return obs[:,:-1], done
    
    def step(self, action):
        obs, reward, done = super().step(action)
        return obs[:,:-1], reward, done


