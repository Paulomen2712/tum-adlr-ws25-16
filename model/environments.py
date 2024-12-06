import gym
from gym.wrappers import RecordVideo
from gym.envs.box2d import LunarLander
import numpy as np
import math

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
    
    def get_environment_shape(self):
        """
            Returns tuple (observatio_space_shape, action_space_shape).
        """
        pass  # Placeholder for environment-specific implementation
    
    def reset(self):
        pass  # Placeholder for environment-specific implementation
    
    def step(self, action):
        pass  # Placeholder for environment-specific implementation
    
    def make_environment_for_recording(self, episode_trigger=lambda _: True):
        """
            Additionally wraps the environment for recording.
        """
        self.env = RecordVideo(self._make_environment(render_mode = 'rgb_array'), video_folder="videos", episode_trigger=episode_trigger)

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
    
    def get_environment_shape(self):
        return self.env.observation_space.shape[0], self.env.action_space.shape[0]
    
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
    
    def reset(self):
        obs, done = super().reset()
        return np.append(obs, self.env.get_wind_mag()), done
    
    def step(self, action):
        obs, reward, done = super().step(action)
        return np.append(obs, self.env.get_wind_mag()), reward, done

class LunarLanderWithWind(LunarLander):
    """
        Custom LunarLander environment with wind turbulence.
    """

    def __init__(self, max_wind_power=15.0, render_mode=None):
        super().__init__(render_mode=render_mode, continuous=True, enable_wind=True, wind_power=np.random.uniform(0, max_wind_power))
        self.max_wind_power = max_wind_power

    def get_wind_mag(self):
        """
            Returns current wind magnitude as calculated by the LunarLander environment.
        """
        return math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )* self.wind_power
        #return self.wind_power

    def reset(self):
        """
            Resets the environment and resamples wind power
        """
        self.wind_power = np.random.uniform(0, self.max_wind_power)
        self.wind_idx = np.random.randint(-9999, 9999)

        return super().reset()