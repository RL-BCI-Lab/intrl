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
from typing import (
    Dict,
    Mapping,
    Optional,
    Sequence,
    Union,
    Callable,
    Tuple
)
from pdb import set_trace

import numpy as np
from stable_baselines3.common.vec_env import (
    DummyVecEnv, 
    is_vecenv_wrapped, 
    VecMonitor,
    is_vecenv_wrapped
)
from stable_baselines3.common.monitor import Monitor
from imitation.data.rollout import (
    AnyPolicy, 
    VecEnv,
    GenTrajTerminationFn,
    flatten_trajectories,
    rollout,
)
from imitation.data import types
from imitation.util.logger import  HierarchicalLogger

from intrl.common.imitation.utils import (
    TransitionsWithFeedback,
    extract_keys_from_trajectory_info,
)


def flatten_trajectories_with_feedback(
    trajectories: Sequence[TransitionsWithFeedback],
    credit_func: Callable = None
) -> TransitionsWithFeedback:
    transitions = flatten_trajectories(trajectories)

    return TransitionsWithFeedback(
        feedbacks=np.concatenate([traj.feedbacks.feedbacks for traj in trajectories]),
        **types.dataclass_quick_asdict(transitions)
    )


def log_rollout_stats(
    trajectories: Sequence[types.TrajectoryWithRew],
    logger: Optional[HierarchicalLogger] = None
) -> Mapping[str, float]:
    """Calculates various stats for a sequence of trajectories.

    Args:
        trajectories: Sequence of trajectories.

    Returns:
        Dictionary containing `n_traj` collected (int), along with episode return
        statistics (keys: `{monitor_,}return_{min,mean,std,max}`, float values)
        and trajectory length statistics (keys: `len_{min,mean,std,max}`, float
        values).

        `return_*` values are calculated from environment rewards.
        `monitor_*` values are calculated from Monitor-captured rewards, and
        are only included if the `trajectories` contain Monitor infos.
    """
    assert len(trajectories) > 0
    out_stats: Dict[str, float] = {"n_traj": len(trajectories)}
    traj_descriptors = {
        "reward": [],
        "cost": [],
        "len": [],
    }
    monitor_ep_returns = []
    
    for t, traj in enumerate(trajectories):
        traj_descriptors['len'].append(len(traj.acts))
        if hasattr(traj, 'rews'):
            total_return = np.sum(traj.rews)
            traj_descriptors['reward'].append(total_return)
            logger.record('reward', total_return)
            logger.record('reward_mean', np.mean(traj.rews))
        else:
            logger.info(f"Trajectory {t} has no 'rews' key.")
        if traj.infos is not None:
            ep_return = traj.infos[-1].get("episode", {}).get("r")
            logger.record("episode", t)
            logger.record("length", len(traj))
            extracted_info = extract_keys_from_trajectory_info(
                trajectory=traj, 
                extract_keys={'cost':'cost'},
                remove=False,
            )
            if 'cost' in extracted_info:
                cost = extracted_info['cost']
                cost_total = np.sum(cost)
                traj_descriptors['cost'].append(cost_total)
                logger.record("cost", cost_total)
                logger.record("cost_mean", np.mean(cost))
            else:
                logger.info(f"Trajectory {t} infos has no 'cost' key.")
                
            logger.dump(step=1)
            
            if ep_return is not None:
                monitor_ep_returns.append(ep_return)
        else:
            logger.info(f"Trajectory {t} has no 'infos' key.")
            
    if monitor_ep_returns:
        # Note monitor_ep_returns[i] may be from a different episode than ep_return[i]
        # since we skip episodes with None infos. This is OK as we only return summary
        # statistics, but you cannot e.g. compute the correlation between ep_return and
        # monitor_ep_returns.
        traj_descriptors["monitor_return"] = np.asarray(monitor_ep_returns)
        # monitor_return_len may be < n_traj when infos is sometimes missing
        out_stats["monitor_return_len"] = len(traj_descriptors["monitor_return"])

    stat_names = ["min", "mean", "std", "max"]
    for desc_name, desc_vals in traj_descriptors.items():
        if len(desc_vals) == 0:
            continue
        for stat_name in stat_names:
            stat_value: np.generic = getattr(np, stat_name)(desc_vals)
            # Convert numpy type to float or int. The numpy operators always return
            # a numpy type, but we want to return type float. (int satisfies
            # float type for the purposes of static-typing).
            out_stats[f"{desc_name}_{stat_name}"] = stat_value.item()

    for v in out_stats.values():
        assert isinstance(v, (int, float))
    return out_stats

class FakeRNG():
    
    @staticmethod
    def shuffle(x, **kwargs):
        """Replicate np.random.shuffle to prevent shuffle from actually shuffling"""
        pass

class RolloutCollector():
    """ Class wrapper for Imitations data module
    
        The goal of this class is to containerize Imitation's data module to all for
        expert demonstration capturing. In the future, if needed, this class can fully
        implement the logic for capturing expert demonstrations instead of wrapping 
        logic provided by Imitation. Currently this is not needed.
        
        Args:
            policy: Can be any of the following:
                1) A stable_baselines3 policy or algorithm trained on the gym environment.
                2) A Callable that takes an ndarray of observations and returns an ndarray
                of corresponding actions.
                3) None, in which case actions will be sampled randomly.
            venv: The vectorized environments.
            sample_until: End condition for rollout sampling.
            rng: Random state to use for sampling.
            unwrap: If True, then save original observations and rewards (instead of
                potentially wrapped observations and rewards) by calling `unwrap_traj()`.
            exclude_infos: If True, then exclude `infos` from pickle by setting
                this field to None. Excluding `infos` can save a lot of space during
                pickles.
            verbose: If True, then print out rollout stats before saving.
        Returns:
            Sequence of trajectories, satisfying `sample_until`. Additional trajectories
            may be collected to avoid biasing process towards short episodes; the user
            should truncate if required.

    """
    def __init__(
        self,
        policy: AnyPolicy, 
        env: VecEnv, 
        sample_until: GenTrajTerminationFn,
        *,
        deterministic_policy: bool = False,
        env_seed: int = None,
        seed: Union[int, None] = None,
        unwrap: bool = True,
        exclude_infos: bool = True,
        verbose: bool = False,
    ):
        self.policy = policy
        self.env = DummyVecEnv([lambda: env]) if not isinstance(env, VecEnv) else env
        self.sample_until = sample_until
        self.deterministic_policy = deterministic_policy
        self.env_seed = env_seed
        # NOTE: rollout() will shuffle trajectories once collected, this is typically
        #       not needed when collecting evaluations or human trajectories.
        self.rng = FakeRNG if seed is None else np.random.default_rng(seed)
        self.unwrap = unwrap
        self.exclude_infos = exclude_infos
        self.verbose = verbose
        
        is_monitor_wrapped = is_vecenv_wrapped(self.env, VecMonitor) or self.env.env_is_wrapped(Monitor)[0]
        if not is_monitor_wrapped and warn:
            warnings.warn(
                "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
                "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
                "Consider wrapping environment first with ``Monitor`` wrapper.",
                UserWarning,
            )
    
    def policy_to_callable(self):
        """ Converts policy into a function using duck typing."""
        if self.policy is None:
            def get_actions(
                observations: Union[np.ndarray, Dict[str, np.ndarray]],
                states: Optional[Tuple[np.ndarray, ...]],
                episode_starts: Optional[np.ndarray],
            ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
                acts = [venv.action_space.sample() for _ in range(len(observations))]
                return np.stack(acts, axis=0), None
        elif hasattr(self.policy, 'predict'):
            def get_actions(
                observations: Union[np.ndarray, Dict[str, np.ndarray]],
                states: Optional[Tuple[np.ndarray, ...]],
                episode_starts: Optional[np.ndarray],
            ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
                (acts, states) = self.policy.predict(  
                    observations,
                    state=states,
                    episode_start=episode_starts,
                    deterministic=self.deterministic_policy,
                )
                return acts, states
        elif callable(self.policy):
            # When a policy callable is passed, by default we will use it directly.
            # We are not able to change the determinism of the policy when it is a
            # callable that only takes in the states.
            if self.deterministic_policy:
                raise ValueError(
                    "Cannot set deterministic_policy=True when policy is a callable, "
                    "since deterministic_policy argument is ignored.",
                )
            get_actions = self.policy
        else:
            raise TypeError(
                "Policy must be None, duck typed to follow stable-baselines policy "
                f"or algorithm predict() methods, or a Callable, got {type(policy)} instead"
            )
            
        return get_actions
    
    def __call__(self) -> Sequence[types.TrajectoryWithRew]:
        self.env.seed(seed=self.env_seed)

        trajectories = rollout(
            policy=self.policy_to_callable(),
            venv=self.env,
            sample_until=self.sample_until,
            deterministic_policy=self.deterministic_policy,
            rng=self.rng,
            unwrap=self.unwrap,
            exclude_infos=self.exclude_infos,
            verbose=self.verbose
        )
        
        return trajectories
        


        