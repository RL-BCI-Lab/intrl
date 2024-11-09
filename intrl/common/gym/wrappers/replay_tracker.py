import warnings
from typing import Optional, Tuple
from pdb import set_trace

import cv2
import numpy as np
import gymnasium as gym

try:
    import mujoco
except ModuleNotFoundError:
    pass


class ReplayTracker(gym.Wrapper):
    """ Track the rendered state 
    
        Stores the rendered state regardless of if the environment is
        image based or not. This can be useful for replaying a task after it
        has been completed. 
    """
    _allowed_modes = ['rgb_array']

    def __init__(self, env):
        super().__init__(env)
        if self.render_mode not in self._allowed_modes:
            msg = "render_mode is invalid. Human observations are unlikely to be stored. " \
                  f"Set render_mode to one of the following modes {self._allowed_modes} to " \
                  "properly capture rendered states."
            warnings.warn(msg)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['render'] = self.get_render()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['render'] = self.get_render()
        # tpr = self.env.unwrapped.task.viewer._time_per_render
        # print(tpr, 1/tpr)
        return obs, reward, terminated, truncated, info
    
    def get_render(self):
        return self.render()


class MujocoReplayTracker(ReplayTracker):
    _allowed_modes = ['rgb_array', 'human']
    
    def __init__(self, env, resize: Optional[Tuple] = None):
        super().__init__(env)
        self.resize = resize
        self._has_init = False if self.render_mode == 'human' else True

    def init(self):
        self.viewport = self.env.unwrapped.task.viewer.viewport
        self.con = self.env.unwrapped.task.viewer.con
        self._has_init = True
 
    def get_render(self):
        if not self._has_init:
            self.init()

        rgb_arr = np.zeros(
            3 * self.viewport.width * self.viewport.height, dtype=np.uint8
        )
        mujoco.mjr_readPixels(rgb_arr, None, self.viewport, self.con)
        
        img = rgb_arr.reshape(self.viewport.height, self.viewport.width, 3)[::-1, :, :]
        if self.resize is not None:
            img = cv2.resize(img, dsize=self.resize, interpolation=cv2.INTER_CUBIC)
            
        return img