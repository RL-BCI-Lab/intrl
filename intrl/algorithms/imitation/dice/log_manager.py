from typing import (
    Union,
    Mapping,
    List,
    TypeVar,
)
from pdb import set_trace

import tensorflow as tf
from stable_baselines3.common import policies
from imitation.util import logger as imit_logger

Dataclass = TypeVar("Dataclass")

class DICELogManager:
    """Utility class to help logging information relevant to Behavior Cloning."""
    def __init__(
        self, 
        logger: imit_logger.HierarchicalLogger,
        name: Union[str] = None,
        *, 
        networks_to_track: List[str] = None
    ):
        """Create new BC logger.
        Args:
            logger: The logger to feed all the information to.
        """
        self.logger = logger
        self.name = name
        self._tensorboard_step = 0
        self._current_epoch = 0
        self._output_tensorboard = 'tensorboard' in logger.format_strs
        self._format_strs = tuple([fstr for fstr in logger.format_strs if fstr not in ['tensorboard', 'wandb']])
        self.networks_to_track = networks_to_track 
        
    def reset_tensorboard_steps(self):
        self._tensorboard_step = 0

    def log_epoch(self, epoch_number):
        self._current_epoch = epoch_number

    def log_batch(
        self,
        batch_num: int,
        num_samples_so_far: int,
        policy,
        training_metrics: Dataclass,
        rollout_stats: Mapping[str, float],
    ):
        self.record("epoch", self._current_epoch)
        self.record("batch", batch_num)
        self.record("samples_so_far", num_samples_so_far)
        for k, v in training_metrics.__dict__.items():
            name = f"{k}"
            if v is None or (tf.is_tensor(v) and len(v.shape) >= 1):
                self.record(name, v, exclude=('csv',))
            else:
                self.record(name, float(v))
            
        # if self._output_tensorboard:
        #     self._log_weights_and_grads(policy=policy)
            
        for k, v in rollout_stats.items():
            if "return" in k and "monitor" not in k:
                self.record("rollout/" + k, v)
        self.logger.dump(self._tensorboard_step)
        self._tensorboard_step += 1

    # def _log_weights_and_grads(self, policy):
    #     for network_name in self.networks_to_track:
    #         network = rgetattr(policy, network_name)
    #         for n, p in network.named_parameters():
    #             n = f"{network_name}.{n}"
    #             self.record(f"params/{n}", p.cpu(), exclude=self._format_strs)
    #             if hasattr(p, 'grad') and p.grad is not None:
    #                 self.record(f"grads/{n}", p.grad.cpu(), exclude=self._format_strs)
    
    def record(self, name, value, exclude=None):
        if self.name is not None:
            self.logger.record(f'{self.name}/{name}', value, exclude=exclude)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_logger"]
        return state
