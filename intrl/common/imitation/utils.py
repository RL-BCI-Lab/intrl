"""
MIT License

Copyright (c) 2019-2022 Center for Human-Compatible AI and Google LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import dataclasses
import os
import warnings
import zipfile
import io
from dataclasses import field
from collections import Iterable
from pathlib import Path
from typing import (
    Optional,
    Mapping,
    Sequence,
    Iterable,
    List,
    Union,
    Dict,
    cast, 
    Any
)
from pdb import set_trace
from collections.abc import Iterable

import numpy as np
import gymnasium as gym
import torch.utils.data as th_data
from imitation.data import types
from imitation.util.util import parse_path
from imitation.data.types import AnyPath, Trajectory, TrajectoryWithRew, Transitions

from intrl.common.logger import logger
from intrl.common.utils import load_files, get_one_hot

DATA_KEYS = ['obs', 'acts', 'terminal', 'infos', 'rews', 'feedbacks']


def transitions_collate_fn(
    batch: Sequence[Mapping[str, np.ndarray]],
) -> Mapping[str, types.AnyTensor]:
    """ Custom `torch.utils.data.DataLoader` collate_fn for `TransitionsMinimal`.

        This builds upon Imitation's types.transitions_collate_fn() function but now
        allows for rewards and feedbacks to be accounted for if needed.

        Args:
            batch: The batch to collate.

        Returns:
            A collated batch. Uses Torch's default collate function for everything
            except the "infos" key. For "infos", we join all the info dicts into a
            list of dicts. (The default behavior would recursively collate every
            info dict into a single dict, which is incorrect.)
    """
    results = types.transitions_collate_fn(batch)
    batch_feedbacks_and_rewards = [
        {k: np.array(v) for k, v in sample.items() if k in ["feedbacks", "rews"]}
        for sample in batch
    ]

    fbs_rews = th_data.dataloader.default_collate(batch_feedbacks_and_rewards)
    if 'rews' in fbs_rews:
        results['rews'] = fbs_rews['rews']
    
    if 'feedbacks' in fbs_rews:
        results['feedbacks'] = fbs_rews['feedbacks']

    return results


def make_data_loader_from_flattened(
    transitions,
    batch_size: int,
    data_loader_kwargs: Optional[Mapping[str, Any]] = None,
) -> Iterable[types.TransitionMapping]:
    """ Converts demonstration data to Torch data loader.

        Adapts Imitation's make_data_loader() to take in custom collate function and
        skips flattening the data and assumes this is done already.
        
        Args:
            transitions: Transitions expressed directly as a `types.TransitionsMinimal`
                object, a sequence of trajectories, or an iterable of transition
                batches (mappings from keywords to arrays containing observations, etc).
            batch_size: The size of the batch to create. Does not change the batch size
                if `transitions` is already an iterable of transition batches.
            data_loader_kwargs: Arguments to pass to `th_data.DataLoader`.

        Returns:
            An iterable of transition batches.

        Raises:
            ValueError: if `transitions` is an iterable over transition batches with batch
                size not equal to `batch_size`; or if `transitions` is transitions or a
                sequence of trajectories with total timesteps less than `batch_size`.
            TypeError: if `transitions` is an unsupported type.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size={batch_size} must be positive.")

    if isinstance(transitions, types.TransitionsMinimal):
        if len(transitions) < batch_size:
            raise ValueError(
                f"Number of transitions in `demonstrations` {len(transitions)} "
                f"is smaller than batch size {batch_size}.",
            )

        kwargs = {
            "shuffle": True,
            "drop_last": True,
            "collate_fn": transitions_collate_fn,
            **(data_loader_kwargs or {}),
        }

        return th_data.DataLoader(
            transitions,
            batch_size=batch_size,
            **kwargs,
        )
    else:
        raise TypeError(f"`transitions` unexpected type {type(transitions)}")
    
    
def extract_keys_from_trajectory_info(
    trajectory: Trajectory, 
    extract_keys: Dict[str, str] = None,
    remove: bool = True,
) -> Dict[str, list]:
    """ Extracts keys from infos array of dictionaries.
        
        WARNING: This function will modify the trajectory.infos dictionary!

        Args:
            trajectory: Imitation Trajectory class
            
            extract_keys:  dictionary mapping info key name to a potentially
                new extracted key name.
                
                Example:
                    extract_keys = {
                        'TimeLimit.truncated':'truncated',
                        'lives':'lives'
                    }
        Return:
            Dictionary with extracted key name mapped to its extracted data.
    """
   
    
    extracted_info = {}
    for info in trajectory.infos:
        extract_keys_ = {i:i for i in info.keys()} if extract_keys is None else extract_keys
        for info_key, extract_key in extract_keys_.items():
            if info_key in info:
                fn = info.pop if remove else info.get
                if extract_key not in extracted_info:
                    extracted_info[extract_key] = [fn(info_key)]
                else:
                    extracted_info[extract_key].append(fn(info_key))
    
    # Warn that there are missing keys if none found.
    ek_values = np.array(list(extract_keys.values()))
    missing_keys = [ek not in extracted_info for ek in ek_values]
    if np.any(missing_keys):
        msg = f"Failed to extract {ek_values[missing_keys]} key(s) from infos."
        warnings.warn(msg)

    return extracted_info


def remap_trajectory_actions(trajectories, action_map):
    """ Maps unique action values to indexes for each trajectory
    
        Example:
            Maps -1 to index 2 for the 1st and 2nd action dimensions
            
            action_map = np.array[[
                [0, 1, -1],
                [0, 1, -1],
            ]
    
    """
    for i, traj in enumerate(trajectories):
        actions = remap_transition_actions(traj, action_map)
        trajectories[i] = dataclasses.replace(traj, acts=actions)
    return trajectories

def remap_transition_actions(transitions, action_map):
    """ Maps unique action values to indexes for each transition
    
        Example:
            Maps -1 to index 2 for the 1st and 2nd dimensions
            
            action_map = np.array[[
                [0, 1, -1],
                [0, 1, -1],
            ]
    
    """
    acts = transitions.acts.copy()
    # Loop over the action dimensions
    for dim, actions in enumerate(action_map):
        # Loop over the actions in the dimension
        for idx, a in enumerate(actions):
            acts[:, dim][transitions.acts[:, dim] == a] = idx
    return acts.astype(int)


def noisify_trajectory_actions(
    trajectories, 
    action_space, 
    epsilon=0.5, 
    seed=None, 
    verbose=False
):
    space_type = type(action_space)
    if isinstance(action_space, gym.spaces.discrete.Discrete):
        return noisify_discrete_actions(
            trajectories=trajectories, 
            action_space=action_space, 
            epsilon=epsilon, 
            seed=seed, 
            verbose=verbose
        )
    else:
        msg = f"`action_space` of type {space_type} is not a compatible."
        raise ValueError(msg)


def noisify_discrete_actions(
    trajectories, 
    action_space, 
    epsilon=0.5, 
    seed=None, 
    verbose=False
):
    if not isinstance(action_space, gym.spaces.discrete.Discrete):
        msg = f"`action_space` of type {action_space} is not a compatible."
        raise ValueError(msg)
    
    log = logger.info if logger.CURRENT else print
    action_space.seed(seed)
    rng = np.random.default_rng(seed)
    for t, traj in enumerate(trajectories):
        if verbose: log(f"Trajectory {t}\n\tOriginal:{traj.acts}")
        n_acts = len(traj.acts)
        act_idx = np.arange(n_acts)
        noisy_act_idx = rng.choice(act_idx, size=int(epsilon * n_acts), replace=False)
        for i in noisy_act_idx:
            curr_act = traj.acts[i]
            one_hot = get_one_hot(np.array([curr_act]), action_space.n).astype(np.int8).flatten()
            new_act = action_space.sample(mask=1-one_hot)
            traj.acts[i] = new_act
        if verbose: log(f"\tNoisy:{traj.acts}")
       
    # Unset seed
    action_space.seed(None)
    
    return trajectories


def _feedback_validation(feedbacks: np.ndarray, acts: np.ndarray):
    if feedbacks.shape != (len(acts),):
        raise ValueError(
            "feedbacks must be 1D array, one entry for each action: "
            f"{feedbacks.shape} != ({len(acts)},)",
        )

    if not np.issubdtype(feedbacks.dtype, np.number):
        raise ValueError(f"feedbacks dtype {feedbacks.dtype} not a number")


@dataclasses.dataclass(frozen=True, eq=False)
class Feedbacks():
    feedbacks: np.ndarray
    time: Optional[np.ndarray] = None
    attrs: Optional[Dict] = field(default_factory=dict)
    

@dataclasses.dataclass(frozen=True, eq=False)
class TrajectoryWithFeedback(Trajectory):
    """ A `Trajectory` that additionally includes feedback information.
    
        Attributes:
            feedbacks: Has shape (trajectory_len, ). dtype int.
    """

    feedbacks: Feedbacks
    
    def __post_init__(self):
        """Performs input validation, including feedbacks."""
        super().__post_init__()
        _feedback_validation(self.feedbacks.feedbacks, self.acts)
   
        
@dataclasses.dataclass(frozen=True, eq=False)
class TrajectoryWithRewFeedback(TrajectoryWithRew):
    """ A `Trajectory` that additionally includes feedback and reward information.
    
        Attributes:
            feedbacks: Has shape (trajectory_len, ). dtype int.
    """

    feedbacks: Feedbacks
    
    def __post_init__(self):
        """Performs input validation, including feedbacks."""
        super().__post_init__()
        _feedback_validation(self.feedbacks.feedbacks, self.acts)


@dataclasses.dataclass(frozen=True)
class TransitionsWithFeedback(Transitions):
    """ A batch of obs-act-obs-rew-feedback-done transitions.
    
        Attributes:
            feedbacks: Has shape (batch_size, ). dtype int.
    """

    feedbacks: Optional[np.ndarray]

    def __post_init__(self):
        """Performs input validation, including feedbacks."""
        super().__post_init__()
        _feedback_validation(self.feedbacks, self.acts)
