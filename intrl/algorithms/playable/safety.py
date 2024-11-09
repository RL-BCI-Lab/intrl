import warnings
from pdb import set_trace
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import gymnasium as gym
import mujoco
import glfw
from safety_gymnasium.builder import Builder
from safety_gymnasium.agents import Point, Car

from intrl.algorithms.playable import BasePlayablePolicy

class SetGLFWCKeyCallbacks():
    """ Class for appending multiple key callbacks for glfw """
    _callbacks = []

    @classmethod
    def add_callback(cls, window, fn: Callable):
        assert isinstance(fn, Callable)
        cls._callbacks.append(fn)
        glfw.set_key_callback(window, cls.callback)

    @classmethod
    def callback(cls, window, key, scancode, action, mods):
        for cb in cls._callbacks:
            cb(window, key, scancode, action, mods)

      
class MujocoHumanSettingsWrapper(gym.Wrapper):
    """ Set glfw settings for screen when human is playing """

    def __init__(self, env, debug_menu: bool = False):
        super().__init__(env)
        self.debug_menu = debug_menu
        
        self._has_set = False
        self._episode = 0
        self._n_episodes = 0
        self._step = 0
        self._menu = {}
        self.waiting = True
        self.running_menu = False
        self.keep_episodes = []
        
        self._create_menu()
        
    def _create_menu(self):
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        self.add_overlay(topleft, "Recapture epsiode", '[Backspace]')
        self.add_overlay(topleft, "Continue to next epsiode", '[Space]')
        
    def add_overlay(self, gridpos: int, text1: str, text2: str):
        """Overlays text on the scene."""
        if gridpos not in self._menu:
            self._menu[gridpos] = ["", ""]
        self._menu[gridpos][0] += text1 + "\n"
        self._menu[gridpos][1] += text2 + "\n"
    
    def _menu_key_callback(self, window, key, scancode, action, mods):
        if self.running_menu and key == glfw.KEY_BACKSPACE:
            self._episode -= 1
            self.waiting = False
        elif self.running_menu and key == glfw.KEY_SPACE:
            self.keep_episodes.append(self._n_episodes-1)
            self.waiting = False
            
    def _set_settings(self):
        render_mode = self.env.render_mode
        if render_mode == 'human':
            self.env.unwrapped.task._get_viewer(render_mode)
            self.env.unwrapped.task.viewer._hide_menu = True
            self.env.unwrapped.task.viewer._scroll_callback(None, None, 25)
            self.env.unwrapped.task.viewer._render_every_frame = True
            
            window = self.env.unwrapped.task.viewer.window
            SetGLFWCKeyCallbacks.add_callback(
                fn=self._menu_key_callback,
                window=window
            )
            if self.debug_menu:
                SetGLFWCKeyCallbacks.add_callback(
                    fn=self.env.unwrapped.task.viewer._key_callback,
                    window=window
                )
            self._has_set = True
        else:
            msg = "The `MujocoHumanSettingsWrapper` requires `render_mode='human'` to work."
            warnings.warn(msg)
            
    def reset(self, **kwargs):
        if self._has_set:
            self.running_menu = True
            self._run_menu_interface()
            self.running_menu = False
        
        output = self.env.reset(**kwargs)

        if not self._has_set:
            self._set_settings()

        self._episode += 1 # Tracks current episode
        self._n_episodes += 1 # Tracks total episodes
        self._step = 0
        
        return output
    
    def _run_menu_interface(self) -> None:
        """ Loops the menu interface until a response is given """
        viewer = self.env.unwrapped.task.viewer
        self.waiting = True
        while self.waiting:
            glfw.set_window_title(viewer.window, f"Steps: {self._step} Episodes: {self._episode}/{self._n_episodes}")
            if viewer.window is None:
                return
            elif glfw.window_should_close(viewer.window):
                glfw.destroy_window(viewer.window)
                glfw.terminate()
            viewer.viewport.width, viewer.viewport.height = glfw.get_framebuffer_size(
                viewer.window
            )
            mujoco.mjv_updateScene(
                viewer.model,
                viewer.data,
                viewer.vopt,
                mujoco.MjvPerturb(),
                viewer.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                viewer.scn,
            )
            mujoco.mjr_render(viewer.viewport, viewer.scn, viewer.con)
            for gridpos, [t1, t2] in self._menu.items():
                mujoco.mjr_overlay(
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    gridpos,
                    viewer.viewport,
                    t1,
                    t2,
                    viewer.con,
                )
            glfw.swap_buffers(viewer.window)
            glfw.poll_events()
            
    def step(self, action):
        output = self.env.step(action)
        self._step += 1
        return output


class SafetyPlayablePolicy(BasePlayablePolicy):
    """ Hijacks the Safety Gymnasium and Gymnasium glfw viewer to get keyboard input actions
    
        keys_to_action = dict(
            <key string>=dict(
                action=<value to use for action>, 
                op=<python operator>, 
                index=<action index to change>
            )
        )
    
    """
    def __init__(self,env, keys_to_action):
        self.env = env
        assert isinstance(env.unwrapped, Builder)
        assert isinstance(env.unwrapped.task.agent, (Point, Car))
        self.keys_to_action = keys_to_action

        self.window = None
        self._pressed_keys = set()
        self._has_init = False
        
    @property
    def running(self):
        return not bool(glfw.window_should_close(self._window))

    def __call__(
        self,
        observations: np.ndarray = None,
        states: Optional[Tuple[np.ndarray, ...]] = None,
        episode_starts: Optional[np.ndarray] = None,
    ):
        if not self._has_init:
            self.init()

        action = np.zeros(self.env.action_space.shape)
        for pressed_key in self._pressed_keys:
            for target_key, key_info in self.keys_to_action.items():
                if pressed_key == self.get_glfw_key(target_key):
                    index = key_info['index'] if key_info['index'] is not None else slice(None)
                    op = key_info['op']
                    action_to_take = key_info['action']
                    if op is None:
                        action[index] = action_to_take
                    else:
                         action[index] =  op(action[index], action_to_take)
                    break
                
        return np.array([action]), None

    def init(self):
        self.env.unwrapped.task._get_viewer(self.env.render_mode)
        self._window = self.env.unwrapped.task.viewer.window
        SetGLFWCKeyCallbacks.add_callback(fn=self._action_key_callback, window=self._window)
        self._has_init = True
    
    def _action_key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self._pressed_keys.add(key)
        elif action == glfw.RELEASE:
            self._pressed_keys.remove(key)
      
    def get_glfw_key(self, key: str):
        return getattr(glfw, f'KEY_{key.upper()}')
  