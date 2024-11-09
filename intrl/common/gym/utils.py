from typing import Callable
from copy import deepcopy
from pdb import set_trace

import numpy as np
import gymnasium as gym
import ale_py
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv 
from ale_py import ALEInterface

def get_n_actions(action_space):
    if isinstance(action_space, gym.spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, gym.spaces.Discrete):
        return int(action_space.n)
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return list(action_space.nvec)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")

def get_ALE(ale_env):
    try:
        env_module = getattr(ale_py.roms, ale_env)
        return env_module
    except AttributeError:
        # msg = f'The passed environment {ale_env} is not an ALE environment.'
        return False

def make_env(env_name, **make_kwargs):
    """ Returns a Gym environment """
    env_name_parts = env_name.split('/')
    if env_name_parts[0] == 'ALE':
        env_name = env_name_parts[1].split('-')[0]
    else:
        env_name = env_name_parts[0]
    
    # Check if game is an ALE module
    env_module = get_ALE(env_name)
    if env_module:
        ale = ALEInterface()
        ale.loadROM(env_module)

    return gym.make(env_name, **make_kwargs)
    
def wrap_env(env, wrappers):
    """ Applies Gym wrappers to environment 
    
        For more info see Gym docs: https://github.com/openai/gym/tree/master/gym/wrappers
        
        wrappers = [
            {'wrapper':, 'args': [], 'kwargs': {}}
        ]
        
        OR
        
        wrappers = [
            Object, Object, Object
        ]
    """
    for w in wrappers:
        if isinstance(w, Callable): 
            env = w(env)
        else:
            kwargs = w['kwargs'] if 'kwargs' in w else {}
            kwargs = kwargs if kwargs is not None else {}
            args = w['args'] if 'args' in w else []
            args = args if args is not None else []
            env = w['wrapper'](env, *args, **kwargs)
    return env


def vectorize_env(env, count: int = 1) -> VecEnv:
    """ Duplicates environment to create dummy vectorized environments """
    return DummyVecEnv([(lambda e: lambda: deepcopy(e))(env) for _ in range(count)])
