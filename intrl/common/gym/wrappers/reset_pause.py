import time

import gymnasium as gym
        
class ResetPause(gym.Wrapper):
    """ Pause briefly between resets """

    def __init__(self, env, duration=1):
        self.duration = duration
        super().__init__(env)

    def reset(self, **kwargs):
        time.sleep(self.duration)
        return self.env.reset(**kwargs) 
