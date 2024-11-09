from pdb import set_trace

import gymnasium as gym
import numpy as np

class ImageNormalization(gym.core.ObservationWrapper):
    def __init__(self, env, dtype=np.float32):
        super().__init__(env)
        self.dtype = dtype
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=dtype)
        
    def observation(self, observation):
        observation = (observation / 255.0).astype(self.dtype)
        return observation
    