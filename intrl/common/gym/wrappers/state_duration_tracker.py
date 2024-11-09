import time
from pdb import set_trace

import gymnasium as gym
import numpy as np
        
class StateDurationTracker(gym.Wrapper):
    """ Track the duration of each state """

    def __init__(self, env):
        super().__init__(env)

    # def reset(self, **kwargs):
    #     obs, info = self.env.reset(**kwargs) 
    #     return obs, info
    
    def step(self, action):
        state_start_time = time.time()
        step_output = self.env.step(action)
        state_end_time = time.time()
        
        step_output[-1]['state_start'] = state_start_time
        step_output[-1]['state_end'] = state_end_time
        
        return step_output
