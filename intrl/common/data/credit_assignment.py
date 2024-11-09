from dataclasses import replace
from typing import Callable, Dict
from pdb import set_trace

import numpy as np


def feedback_credit(
    trajectories,
    mapping_func: Callable, 
    **kwargs: Dict
):
    """ Updates trajectories feedbacks dataclass
    
        This can only be done using dataclass replace() method.
    
    """
    for t in range(len(trajectories)):
        trajectories[t] = replace(
            trajectories[t],
            feedbacks=replace(
                trajectories[t].feedbacks, 
                feedbacks=mapping_func(trajectories[t], **kwargs))
        )
    return trajectories


def soft_map_to_state_time(
    trajectory,
    time_key: str = 'time', 
    map_length: float = 0.3
):
    
    fbs = trajectory.feedbacks
    locs = np.where(fbs.feedbacks != 0)[0]
    time = getattr(fbs, time_key)
    credit = np.zeros(fbs.feedbacks.shape)
   
    for l in locs:
        end = time[l]
        start = end - map_length
        credit_locs = np.logical_and(start<=time, end >= time) 
        credit[credit_locs] += fbs.feedbacks[l]
    return credit