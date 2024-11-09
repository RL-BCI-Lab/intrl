from pdb import set_trace
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import Env

class BasePlayablePolicy(ABC):
    def __init__(
        self,
        env: Env,

    ):
        """Wraps an environment with a dictionary of keyboard buttons to action and if to zoom in on the environment.

        Args:
            env: The environment to play
            keys_to_action: The dictionary of keyboard tuples and action value
            zoom: If to zoom in on the environment render
        """
        if env.render_mode not in {"rgb_array", "rgb_array_list"}:
            raise ValueError(
                "PlayableGame wrapper works only with rgb_array and rgb_array_list render modes, "
                f"but your environment render_mode = {env.render_mode}."
            )
        self.env = env
        
    @property
    @abstractmethod
    def running(self):
        pass
    
    def _get_video_size(self, zoom: Optional[float] = None) -> Tuple[int, int]:
        rendered = self.env.render()
        if isinstance(rendered, List):
            rendered = rendered[-1]
        assert rendered is not None and isinstance(rendered, np.ndarray)
        video_size = (rendered.shape[1], rendered.shape[0])

        if zoom is not None:
            video_size = (int(video_size[0] * zoom), int(video_size[1] * zoom))

        return video_size


    @abstractmethod
    def __call__(
        self,
        observations: np.ndarray,
        states: Optional[Tuple[np.ndarray, ...]],
        episode_starts: Optional[np.ndarray],
    ) -> List[int]:
        """ Construct this method to determine logic for playable game loop
        
            Returns:
                A list of actions
        """
        pass