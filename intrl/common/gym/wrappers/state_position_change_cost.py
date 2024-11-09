
from pdb import set_trace

import gymnasium as gym
import numpy as np

class StatePositionChangeCost(gym.Wrapper):
    def __init__(
        self, 
        env,
        obs_dim: int, 
        steps: int = None, 
        threshold: float = 1, 
        decreasing: bool = True,
        noop: int = None
    ):
        super().__init__(env)
        self.obs_dim = obs_dim
        self._track_obs = []
        self.steps = steps
        self.threshold = threshold
        self.decreasing = decreasing
        self.noop = noop
       

    def reset(self, **kwargs):
        output = self.env.reset(**kwargs)
        self._track_obs = []
        return output
    
    def _monotonic(self, steps):
        start = self._track_obs
        # steps+1 makes list == steps unless steps = len(self._track_obs)
        end = self._track_obs[1:steps] 
        if self.decreasing:
            return [s>=e for s, e in zip(start, end)]
        else: 
            return [s<=e for s, e in zip(start, end)]
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.noop is not None and self.noop != action:
            self._track_obs.append(obs[self.obs_dim])
   
        if terminated or truncated:
            steps = len(self._track_obs) if self.steps is None else self.steps
            # Minus one to compensate for self._monotonic returning list of length n-1
            min_steps = int(self.threshold * (steps-1))
            monotonic = self._monotonic(steps)
            count = np.sum(monotonic)
            if count >= min_steps:
                info['cost'] = 0
            else:
                info['cost'] = 1
            print(f"Cost: {info['cost']} {count} {min_steps}\n{monotonic}")
            print(self._track_obs[:steps-1])
        return obs, reward, terminated, truncated, info