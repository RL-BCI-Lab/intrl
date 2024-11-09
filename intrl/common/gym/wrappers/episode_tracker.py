
from pdb import set_trace

import gymnasium as gym
import numpy as np
        
class EpisodeTracker(gym.Wrapper):
    """ Track the number of steps and episodes"""

    def __init__(self, env):
        super().__init__(env)
        self._episode = 0
        self._step = 0

    def reset(self, **kwargs):
        self._step = 0
        self._episode += 1
        output = self.env.reset(**kwargs) 
        return output
    
    def step(self, action):
        self._step += 1
        step_output = self.env.step(action)
        return step_output
