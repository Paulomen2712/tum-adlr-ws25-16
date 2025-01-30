from env.ant import AntEnv
import numpy as np
from gym import utils

class Ant_Stiffness(AntEnv):
    def __init__(self, min_stiff = 0, max_stiff = 1, **args):
        super().__init__(**args)
        self.min_stiff = min_stiff
        self.max_stiff = max_stiff

    def add_disturbance(self):
        self.model.jnt_stiffness = self.model.jnt_stiffness * 0 + np.random.uniform(self.min_stiff, self.max_stiff)
    
    def get_disturbance_shape(self):
        return 1

    def get_disturbance(self):
        return [self.model.jnt_stiffness[0]]