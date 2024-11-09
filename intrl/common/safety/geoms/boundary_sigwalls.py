from dataclasses import dataclass

import numpy as np
from safety_gymnasium.assets.geoms import Sigwalls

@dataclass
class BoundarySigwalls(Sigwalls):  # pylint: disable=too-many-instance-attributes
    """Collision walls for the boundary of the map.

        This class is used for showing the boundary walls which can not be passed.
    """

    def get_config(self, xy_pos, rot):  # pylint: disable=unused-argument
        """To facilitate get specific config for this object."""
        body = {
            'name': self.name,
            'pos': np.r_[xy_pos, 0.25],
            'rot': 0,
            'geoms': [
                {
                    'name': self.name,
                    'size': np.array([0.05, self.size, 0.3]),
                    'type': 'box',
                    'group': self.group,
                    'rgba': self.color * np.array([1, 1, 1, self.alpha]),
                },
            ],
        }
        if self.index >= 2:
            body.update({'rot': np.pi / 2})
        self.index_tick()
        if self.is_meshed:
            body['geoms'][0].update(
                {
                    'type': 'mesh',
                    'mesh': self.mesh_name,
                    'material': self.mesh_name,
                    'euler': [0, 0, 0],
                },
            )
        return body