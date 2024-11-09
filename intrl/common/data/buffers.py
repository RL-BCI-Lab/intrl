"""
The MIT License

Copyright (c) 2019 Antonin Raffin

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

import os
import warnings
from dataclasses import dataclass, make_dataclass, fields
from typing import Any, Dict, Generator, List, Optional, Union
from copy import deepcopy
from collections import namedtuple
from pdb import set_trace

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from intrl.common.utils import reduce_dims

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None
    
@dataclass(repr=False)
class Samples():
    """ Dataclass used to replicated NameTuple for storing samples sampled from a replay buffer."""
    observations: Union[np.ndarray, torch.Tensor]
    actions: Union[np.ndarray, torch.Tensor]
    next_observations: Union[np.ndarray, torch.Tensor]
    dones: Union[np.ndarray, torch.Tensor]
    rewards: Union[np.ndarray, torch.Tensor]
    
    def __post_init__(self):
        self._fields = list(self.__dict__.keys())
        # Assumes attributes are always of the same storage type
        # either np.ndarray or torch.Tensor
        self._storage_type = type(self.observations)
        if self._storage_type not in (np.ndarray, torch.Tensor):
            raise TypeError(f"Storage type {self._storage_type} can only be np.ndarray or torch.Tensor")
        
    def __getitem__(self, index):
        return [getattr(self, f)[index] for f in self._fields]
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self._storage_type})"
    
    def __len__(self):
        return len(self.observations)

    def to_tensor(self, device='cpu'):
        # Always recast to make sure same devices are used.
        if self._storage_type is torch.Tensor:
            [setattr(self, f, getattr(self, f).to(device)) for f in self._fields]
            return self
        elif self._storage_type is not np.ndarray:
            raise TypeError(f"Samples storage {self._storage_type} is not of type np.ndarray")
        [setattr(self, f, torch.as_tensor(getattr(self, f)).to(device)) for f in self._fields]
        self._storage_type = torch.Tensor
        return self
        
    def to_numpy(self):
        def get_numpy(f):
            f = getattr(self, f)
            return f.cpu().numpy() if not f.requires_grad else f.detach().cpu().numpy() 
        if self._storage_type is np.ndarray:
            return
        elif self._storage_type is not torch.Tensor:
            raise TypeError(f"Samples storage {self._storage_type} is not of type torch.Tensor")
        [setattr(self, f, get_numpy(f)) for f in self._fields]
        self._storage_type = np.ndarray
        return self

    def concat(self, samples):
        assert self._storage_type is samples._storage_type
        def numpy_concat(f):
            self_f = getattr(self, f)
            samples_f = getattr(samples, f)
            return np.vstack([self_f, samples_f])
        
        def torch_concat(f):
            self_f = getattr(self, f)
            samples_f = getattr(samples, f)
            return torch.vstack([self_f, samples_f])
        
        if self._storage_type is np.ndarray:
            [setattr(self, f, numpy_concat(f)) for f in self._fields if f in samples._fields]
        else:
            [setattr(self, f, torch_concat(f)) for f in self._fields if f in samples._fields]
        return self
    
    def reduce_dims(self, dims, exclude=None, include=None, reshape_order='C'):
        _reduce_dims_func = lambda f: setattr(self, f, reduce_dims(getattr(self, f), dims, reshape_order=reshape_order))
        if exclude is not None and include is not None:
            msg = "Can only specify arguments for exclude OR include not both."
            raise ValueError(msg)
        elif exclude is None and include is None:
            [_reduce_dims_func(f) for f in self._fields]
        elif exclude:
            [_reduce_dims_func(f) for f in self._fields if f not in exclude]
        elif include:
            [_reduce_dims_func(f) for f in self._fields if f in include]
        return self
    
class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        keep_infos_keys: dict = None,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size=buffer_size, 
            observation_space=observation_space, 
            action_space=action_space, 
            device=device, 
            n_envs=n_envs
        )
 
        self.debug=False
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage
        
        # Shape (timesteps, envs, frames, color, height, width)
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        self.infos = {}
        if keep_infos_keys is not None:
            for key, type in keep_infos_keys.items():
                self.infos[key] = np.zeros((self.buffer_size, self.n_envs), dtype=type)
                
        self._track_info_keys = keep_infos_keys
        self.sample_class = self._get_sample_class(keep_infos_keys)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes \
            + self.actions.nbytes \
            + self.rewards.nbytes \
            + self.dones.nbytes \
            + sum([i.nbytes for i in self.infos.values()])

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            print("Required Memory: {:.2f}GB Available Memory: {:.2f}GB".format(total_memory_usage/1e9, mem_available/1e9))
            if total_memory_usage > mem_available:
                total_memory_usage /= 1e9 # Convert to GB
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def __len__(self):
        return self.buffer_size if self.full else self.pos
    
    def _get_sample_class(self, add_attributes=None):
        add_attributes = [] if add_attributes is None else add_attributes
        return make_dataclass('ReplaySamples', add_attributes, bases=(Samples,), repr=False)
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        info: List[Dict[str, Any]] = None,
    ) -> None:
        """ Add samples to replay buffer.
        
        
            NOTE: .copy() seems unneeded as np.shares_memory() and np.may_share_memory()
                  seem to always return False when not using .copy(). np.array() should
                  always copy by default.
                  Ref: https://github.com/DLR-RM/stable-baselines3/issues/112
        """

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        # Check infos to see if there is any data that needs to be tracked.
        if info is not None and len(self.infos) != 0:
            for key, _ in self.infos.items():
                data = []
                for env_idx, i in enumerate(info):
                    if key in i:
                        data.append(i[key])
                    else:
                        err = f"The key {key} was not detected in infos. Make sure infos " \
                            f"contains this key or remove it from keep_infos_keys when " \
                            f"initializing ReplayBuffer."
                        raise ValueError(err)
                if len(data) != 0:
                    self.infos[key][self.pos] = np.array(data).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self,
        batch_size: int, 
        random: bool = True,
        all_envs: bool = False,
        combined: bool = False,
        device: str = None
    ) -> Samples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        :param batch_size number samples to draw from buffer
        :random randomly sample from buffer, otherwise will take the latest samples equivalent to the batch_size.
        :param all_envs sample from all environments, otherwise will randomly sample from environments
        If `all_envs = False` and `random = False`, the last samples will be randomly selected
        from all the environments. 
        If `all_envs = True` and `random = False`, the last samples from all the environments 
        will be used.
        :param combined combines samples from replay (batch_size - 1) with most recent sample.
        Can only be used when randomly sampling. If `random = False`, the latest sample is 
        already included in the batch.
        See https://arxiv.org/abs/1712.01275 for combined sampling theory.
        :param device overrides default device specified by class.
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        device = device if device is not None else self.device
        
        assert random or all_envs, "Both random and all_envs can not be False."
        
        def latest_samples(batch_size):
            if upper_bound - batch_size < 0:
                err = f"batch_size {batch_size} can not be greater than upper bound {upper_bound} when using non-random sample."
                raise ValueError(err)
  
            if self.full:
                # Get the latest indices without including self.pos
                batch_inds = (np.arange(self.buffer_size - batch_size, self.buffer_size) + self.pos) % self.buffer_size
            else:
                # No wrapping is ever possible as buffer has never been filled
                batch_inds = np.arange(self.pos - batch_size, self.pos)
                
            return batch_inds
    
        if not self.optimize_memory_usage:
            if random:
                # self.pos is not included as it is assumed to have just been iterated
                # after adding new sample. Once full, pos can be included as it isnt 
                # overwritten, unlike in memory optimization.
                batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            else:
                batch_inds = latest_samples(batch_size)
        else:
            if random:
                if self.full:
                    # Start at 1 to prevent same index as self.pos from being drawn
                    batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
                else:
                    # End at self.pos to prevent same index as self.pos from being generated
                    batch_inds = np.random.randint(0, self.pos, size=batch_size)
            else:
                batch_inds = latest_samples(batch_size)

        # Replaces last index with the current sample index if it hasn't already been selected
        if combined and random:
            current_ind = (self.pos-1) % self.buffer_size
            if current_ind not in batch_inds:
                batch_inds[-1] = (self.pos-1) % self.buffer_size
            # TODO: REMOVE debug check
            assert current_ind in batch_inds
        
        samples = self._get_samples(batch_inds, all_envs=all_envs)
      
        # Always reduce environment dimensions when all_env sampling is used
        # 1st environment samples are fist, 0 to batch_size. 2nd environments
        # samples are next, batch_size to batch_size*2.
        if all_envs:
            samples.reduce_dims([[0, 1]], reshape_order='F')

        return samples.to_tensor(device)

    def _get_samples(
        self, 
        batch_inds: np.ndarray, 
        all_envs: bool = False,
    ) -> Samples:
        if all_envs:
            env_inds = slice(None)
        else:
            env_inds = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_inds, :]
        else:
            next_obs = self.next_observations[batch_inds, env_inds, :]

        data = (
            self.observations[batch_inds, env_inds, :],
            self.actions[batch_inds, env_inds, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_inds] * (1 - self.timeouts[batch_inds, env_inds]))[..., None],
            self.rewards[batch_inds, env_inds][..., None],
        )
        # Append tracked info data, if empty, nothing will happen
        data += tuple([info_array[batch_inds, env_inds] for _, info_array in self.infos.items()])

        return self.sample_class(*data)