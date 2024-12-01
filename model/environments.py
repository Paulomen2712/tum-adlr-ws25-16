import gym
from gym.wrappers import RecordVideo
class LunarContinuous():
    """ OpenAi Lunar Continuous Environment. """

    def __init__(self, should_render = False):
        """
            Initialize parameters.

            Parameters:
                render_mode (string): mode to render the environment
        """
        self.render_mode = "human"  if should_render else None

    def make_environment(self):
        """
            Makes a new environment.
        """
        return gym.make('LunarLanderContinuous-v2', render_mode = self.render_mode)
    
    def make_environment_for_recording(self, episode_trigger=lambda x: True):
        return RecordVideo(gym.make('LunarLanderContinuous-v2', render_mode = 'rgb_array'), video_folder="videos", episode_trigger=episode_trigger)