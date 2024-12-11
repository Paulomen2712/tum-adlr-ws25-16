import gym
from gym.wrappers import RecordVideo
from gym.envs.box2d import LunarLander
from env.lunar_lander import LunarLanderWithWind
import numpy as np
import yaml

class Environment(gym.Wrapper):
    """ Base class for wrapping environments. """
    def __init__(self, **args):
        """
            Initialize parameters.
        """
        self.env = self._make_environment(**args)
        super().__init__(self.env)

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

    def get_observation_shape(self):
        pass

    def get_acition_shape(self):
        pass
    
    def reset(self):
        pass  # Placeholder for environment-specific implementation
    
    def step(self, action):
        pass  # Placeholder for environment-specific implementation
    
    def make_environment_for_recording(self, episode_trigger=lambda _: True):
        """
            Additionally wraps the environment for recording.
        """
        self.env = RecordVideo(self._make_environment(render_mode = 'rgb_array'), video_folder="videos", episode_trigger=episode_trigger)

    def load_hyperparameters(self):
        with open(self._get_config_path(), 'r') as f:
            return yaml.safe_load(f)

class LunarContinuous(Environment):
    """ OpenAi Lunar Continuous Environment Wrapper. """

    def __init__(self, **args):
        """
            Initialize parameters.

            Parameters:
                render_mode (string): mode to render the environment
        """
        super().__init__(**args)

    def _make_environment(self, **args):
        return LunarLander(continuous=True, **args)
    
    def _get_config_path(self):
        return "./configs/LunarLander.yaml"
    
    def get_environment_shape(self):
        return self.env.observation_space.shape[0], self.env.action_space.shape[0]
    
    def get_observation_shape(self):
        return self.env.observation_space.shape

    def get_acition_shape(self):
        return self.env.action_space.shape
    
    def reset(self):
        obs, _ = self.env.reset()
        done = False # done is always false after reset
        return obs, done
    
    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated | truncated
        return obs, reward, done
    
class LunarLanderWithUnknownWind(LunarContinuous):
    """ OpenAi Lunar Continuous Environment Wrapper with wind enabled environment wrapper. """
    def __init__(self, **args):
        super().__init__(**args)

    def _make_environment(self, **args):
        return LunarLanderWithWind(**args)
    
class LunarLanderWithKnownWind(LunarLanderWithUnknownWind):
    """
        OpenAi Lunar Continuous Environment Wrapper that adds the wind to the observation space.
    """
    def __init__(self, **args):
        super().__init__(**args)
    
    def get_environment_shape(self):
        return self.env.observation_space.shape[0] + 1, self.env.action_space.shape[0]
    
    def get_observation_shape(self):
        return (self.env.observation_space.shape[0],)

    def reset(self):
        obs, done = super().reset()
        return np.append(obs, self.env.get_wind_mag()), done
    
    def step(self, action):
        obs, reward, done = super().step(action)
        return np.append(obs, self.env.get_wind_mag()), reward, done

