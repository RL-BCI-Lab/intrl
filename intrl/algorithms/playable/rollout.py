from typing import (
    Sequence,
)
from pdb import set_trace

from imitation.data.rollout import (
    GenTrajTerminationFn,
)
from imitation.data import types
from stable_baselines3.common.env_util import is_wrapped

from intrl.common.data.rollout import RolloutCollector
from intrl.algorithms.playable.pygame import BasePlayablePolicy


class PlayableRolloutCollector(RolloutCollector):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self._validate_policy()
        self.sample_until = self._wrap_sample_until(self.sample_until)
        assert len(self.env.envs) == 1, "Number of envs > 1, PlayableRolloutCollector does not support multiple envs."
        self.keep_trajs = getattr(self.env.envs[0], 'keep_episodes', None)
        
    def _validate_policy(self) -> None:
        msg = f"{self.__class__.__name__} can only use a policy of type BasePlayablePolicy. " \
              f"Try using intrl.common.rollout.RolloutCollector instead."
        assert isinstance(self.policy, BasePlayablePolicy), msg
        
    def _wrap_sample_until(self, sample_until) -> GenTrajTerminationFn:
        def playable_condition(trajs: Sequence[types.TrajectoryWithRew]) -> bool:
            # print(len(trajs), self.keep_trajs)
            trajs = [trajs[i] for i in self.keep_trajs] if self.keep_trajs is not None else trajs
            external_keep_sampling = sample_until(trajs)
            # Stop playing if screen closes (i.e., policy is not running)
            internal_keep_sampling = not self.policy.running
            # if external_keep_sampling or internal_keep_sampling:
            #     print(external_keep_sampling, internal_keep_sampling, self.policy.running)
            return external_keep_sampling or internal_keep_sampling
        
        return playable_condition
    
    def __call__(self):
        trajs = super().__call__()
        try:
            self.env.close()
        except Exception:
            pass
        return [trajs[i] for i in self.keep_trajs] if self.keep_trajs is not None else trajs