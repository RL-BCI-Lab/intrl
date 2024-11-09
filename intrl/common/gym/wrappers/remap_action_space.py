import gymnasium as gym
import numpy as np

class RemapActionSpace(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """ Converts discrete action space indexes back to their original value
    """

    def __init__(self, env, action_map):
        assert isinstance(
            env.action_space, gym.spaces.Box
        ), f"expected Box action space, got {type(env.action_space)}"
        self.action_map = action_map
        self._dims = [len(a) for a in action_map]

        gym.utils.RecordConstructorArgs.__init__(self, action_map=action_map)
        gym.ActionWrapper.__init__(self, env)
        
        self.action_space = gym.spaces.MultiDiscrete(self._dims)


    def action(self, action):
        """ Assumes passed actions are indexes as these will be converted back to a specified value"""
        assert np.all(action >= 0), f"Passed actions are assumed to be indexes but got {action[action < 0]}"
        selected_action = np.take_along_axis(
            self.action_map.T, 
            action.reshape(1, -1).astype(int),
            axis=0
        )
        # print(selected_action.squeeze(), action)
        return selected_action.squeeze()