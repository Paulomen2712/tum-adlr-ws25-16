import gym

class LunarContinuous():
    """ OpenAi Lunar Continuous Environment. """

    def __init__(self, render_mode = 'human'):
        """
            Initialize parameters.

            Parameters:
                render_mode (string): mode to render the environment
        """
        self.render_mode = render_mode

    def make_environment(self):
        """
            Makes a new environment.
        """
        return gym.make('LunarLanderContinuous-v2', render_mode = self.render_mode)