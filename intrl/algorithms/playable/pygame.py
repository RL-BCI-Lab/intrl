"""
The MIT License

Copyright (c) 2016 OpenAI
Copyright (c) 2022 Farama Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from pdb import set_trace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
from pygame.event import Event
from gymnasium.utils.play import display_arr, MissingKeysToAction
from gymnasium import Env
from stable_baselines3.common.env_util import is_wrapped

from intrl.common.gym.wrappers import PyGameRender
from intrl.common.gym.wrappers.pygame_render import PyGameEventHandler
from intrl.algorithms.playable import BasePlayablePolicy

def decode_keys_to_action(env, keys_to_action=None):
    """ Construct a mapping of key codes to actions using key names """
    keycode_to_action = {}
    
    if keys_to_action is None:
        if hasattr(env, "get_keys_to_action"):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert env.spec is not None
            raise MissingKeysToAction(
                f"{env.spec.id} does not have explicit key to action mapping, "
                "please specify one manually"
    )
    assert keys_to_action is not None
    
    for key_combination, action in keys_to_action.items():
        key_code = tuple(
            sorted(ord(key) if isinstance(key, str) else key for key in key_combination)
        )
        keycode_to_action[key_code] = action
    
    return keycode_to_action  


def get_relevant_keys(
    env,
    keys_to_action: Optional[Dict[Tuple[int], int]] = None
) -> set:
    """ Extracts set of keys that have corresponding mappings """
    if keys_to_action is None:
        if hasattr(env, "get_keys_to_action"):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert env.spec is not None
            raise MissingKeysToAction(
                f"{env.spec.id} does not have explicit key to action mapping, "
                "please specify one manually"
            )
    assert isinstance(keys_to_action, dict)
    relevant_keys = set(sum((list(k) for k in keys_to_action.keys()), []))
    return relevant_keys
   
 
class PyGamePlayablePolicy(BasePlayablePolicy):
    """ Implements human controlled policy
    
        This class combines Gym's PlayableGame class and Gym's play() function to allow
        for functionality with Imitation's rollout() function for collecting expert
        data. Here the play() function has largely been converted into the __call__()
        method. Rollout() will thus call this class as if it were a function in order
        to get the next action to use in the current environment.
    """ 
    def __init__(
        self, 
        env: Env, 
        keys_to_action: Optional[Dict] = None,
        noop = None,
        transpose: Optional[bool] = True,
        fps: Optional[int] = 30,
        zoom: Optional[float] = None,
    ):
        assert is_wrapped(env, PyGameRender), "Environment must be wrapped with PyGameRender to work."
        super().__init__(env=env)
        self.noop = noop if noop is not None else 0
        self.keys_to_action = keys_to_action
        self.keycode_to_action = decode_keys_to_action(env, keys_to_action)
        self.relevant_keys = get_relevant_keys(env, self.keycode_to_action)
        self.pressed_keys = []
        self._running = True
        # Events will be handled by PyGameRender
        PyGameEventHandler.event_callbacks.append(self._process_events)
        
    @property
    def running(self):
        return self._running
    
    def __call__(
        self,
        observations: np.ndarray,
        states: Optional[Tuple[np.ndarray, ...]],
        episode_starts: Optional[np.ndarray],
    ) -> List[int]:
      
        # self.process_event()
        # Always select first key input if multiple keys are pressed and
        # don't belong to the keycode action mapping.
        pressed_keys = tuple(sorted(self.pressed_keys))
        action = self.keycode_to_action.get(pressed_keys, self.noop)
        if np.all(action == self.noop) and len(pressed_keys) > 1:
            action = self.keycode_to_action.get(pressed_keys[1:], self.noop)
            
        return np.array([action]), None
    
    def _process_events(self, event): #event: Event):
        """Processes a PyGame event.

        In particular, this function is used to keep track of which buttons are currently pressed
        and to exit the :func:`play` function when the PyGame window is closed.

        Args:
            event: The event to process
        """
        if pygame.event.peek(pygame.QUIT):
            self._running = False
        # for event in pygame.event.get([pygame.KEYDOWN, pygame.KEYUP]):
        if event.type == pygame.KEYDOWN:
            if event.key in self.relevant_keys:
                self.pressed_keys.append(event.key)
        elif event.type == pygame.KEYUP:
            if event.key in self.relevant_keys:
                self.pressed_keys.remove(event.key)
    
