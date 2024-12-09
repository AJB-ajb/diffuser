# utility functions for the empty environment base actions
from minigrid.core.actions import Actions
import numpy as np
from numpy.linalg import norm
import math
from minigrid.envs.empty import EmptyEnv
import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv

import gymnasium as gym
import minigrid_base
from minigrid_base import EnvFeatureCoderBase

class EmptyEnvFC(EnvFeatureCoderBase):
    
    def __init__(self, env_id = "MiniGrid-Empty-16x16-v0") -> None:
        super().__init__()
        self._env = gym.make(env_id) # for calculation of consistency of actions and states
        self._base_env = self._env.unwrapped

    def reset_env_state(self, base_env, raw_obs):
        base_env.agent_start_pos = raw_obs[:2]
        base_env.agent_start_dir = raw_obs[2]
        base_env.reset()
    
    def raw_obs_from_env(self, base_env):
        return np.array([*base_env.agent_pos, base_env.agent_dir])
                
        
class EmptyEnvDiscFC(EmptyEnvFC):
    """
    Feature coder for the empty environment, which translates states and actions to close analogs of their discrete counterparts, i.e. the direction is encoded as discrete pseudo-angle from -1 to 1, and the actions are encoded as one-hot vectors.
    """
    observation_dim = 3
    action_dim = 2
    transition_dim = observation_dim + action_dim
    encoded_actions = [Actions.right, Actions.left, Actions.forward, Actions.done]
    
    def obs_repr_from_env(self, base_env):
        return np.array([*base_env.agent_pos, base_env.agent_dir])
    
    def raw_obs_from_repr(self, continuous_observation):
        return continuous_observation.round().astype(int)
    
    def raw_obs_to_repr(self, raw_obs):
        return raw_obs
    
    def action_to_repr(self, action):
        if action == Actions.right:
            return np.array([1, 0])
        elif action == Actions.left:
            return np.array([-1, 0])
        elif action == Actions.forward:
            return np.array([0, 1])
        else:
            return np.array([0, 0])
    
    def action_from_repr(self, continuous_action):
        """
        Return the approximate action from the continuous action representation.
        """
        # take the slot, where the absolute value is maximal
        slot = np.abs(continuous_action).argmax()
        if np.abs(continuous_action[slot]) < 1e-3:
            return Actions.done
        elif slot == 0:
            return Actions.right if continuous_action[slot] >= 0 else Actions.left
        elif slot == 1:
            return Actions.forward
        
    def calc_state_transition(self, obs_repr1, obs_repr2):
        """
            Calculate the most likely action as well as absolute position and direction change between the two states.
        """
        Δpos = obs_repr2[:2] - obs_repr1[:2]
        Δdir = obs_repr2[2] - obs_repr1[2]

        # circle_dst = min(Δdir % 4, (-Δdir) % 4)
        # go right if the distance in the right direction is smaller
        goes_right = np.argmin([Δdir % 4, (-Δdir) % 4]) == 0

        norm_Δpos = norm(Δpos)
        abs_Δdir = abs(Δdir)

        if norm_Δpos + abs_Δdir < 1e-3:
            action = Actions.done
        elif norm_Δpos > abs_Δdir:
            action = Actions.forward
        else:
            action = Actions.right if goes_right else Actions.left

        return action, (norm_Δpos, abs_Δdir)
        
class EmptyEnvCircFC(EmptyEnvFC):
    """
        Feature coder for the empty environment which encodes the agent direction as (cos(φ), sin(φ)) and the actions as in the `Disc` encoder.
        Here, φ is the angle of the agent direction in the range of [0, 2π), i.e. 2π * agent_dir.

    """
    observation_dim = 4
    action_dim = 2
    transition_dim = observation_dim + action_dim
    encoded_actions = [Actions.right, Actions.left, Actions.forward, Actions.done]

    def raw_obs_to_repr(self, raw_obs):
        angle = 2 * np.pi * raw_obs[2] / 4
        return np.array([*raw_obs[:2], np.cos(angle), np.sin(angle)])
    
    def raw_obs_from_repr(self, obs_repr):
        dir = obs_repr[2:4] / np.linalg.norm(obs_repr[2:4])
        pos = obs_repr[:2].round().astype(int)

        # take the closest direction
        possible_directions  = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        int_dir = np.argmin([np.linalg.norm(dir - d) for d in possible_directions])
        return np.array([*pos, int_dir])
    
    def action_to_repr(self, action):
        if action == Actions.right:
            return np.array([1, 0])
        elif action == Actions.left:
            return np.array([-1, 0])
        elif action == Actions.forward:
            return np.array([0, 1])
        else:
            return np.array([0, 0])
    def action_from_repr(self, action_repr):
        slot = np.abs(action_repr).argmax()
        if np.abs(action_repr[slot]) < 1e-3:
            return Actions.done
        elif slot == 0:
            return Actions.right if action_repr[slot] >= 0 else Actions.left
        elif slot == 1:
            return Actions.forward

    def action_to_repr(self, action):
        if action == Actions.right:
            return np.array([1, 0])
        elif action == Actions.left:
            return np.array([-1, 0])
        elif action == Actions.forward:
            return np.array([0, 1])
        else:
            return np.array([0, 0])
        
    def calc_state_transition(self, obs_repr1, obs_repr2):
        """
            Calculate the most likely action as well as absolute position and direction change between the two states.
            Return the action and norms of position and direction change.
        """
        Δpos = obs_repr2[:2] - obs_repr1[:2]
        Δangle = (np.arctan2(obs_repr2[3], obs_repr2[2]) - np.arctan2(obs_repr1[3], obs_repr1[2])) / (2 * np.pi) * 4
        goes_right = np.argmin([Δangle % 4, (-Δangle) % 4]) == 0

        abs_Δangle = abs(Δangle)
        norm_Δpos = norm(Δpos)
        if norm_Δpos + abs_Δangle < 1e-3:
            action = Actions.done
        elif norm_Δpos > abs_Δangle:
            action = Actions.forward
        else:
            action = Actions.right if goes_right else Actions.left # note that the direction is encoded as (cos(φ), sin(φ)), with the y axis pointing downwards
        
        return action, (norm_Δpos, abs_Δangle)
        
# ------------------- Tests ------------------- #

if __name__ == "__main__":
    def test_env_feature_coder(coder : EnvFeatureCoderBase, base_env):
        for action in coder.encoded_actions:
            assert action == coder.action_from_repr(coder.action_to_repr(action))
            assert np.all(coder.get_transition(base_env, action)[:coder.observation_dim] == coder.obs_repr_from_env(base_env))
        
        obs_repr = coder.obs_repr_from_env(base_env)
        assert np.all(coder.raw_obs_from_repr(obs_repr) == coder.raw_obs_from_repr(obs_repr + 1e-5))
        # test that the difference between sampled state from the encoded state is small
        assert coder.repr_diff(obs_repr) < 1e-10
    
        print("All tests passed.")
    
    env_id = "MiniGrid-Empty-16x16-v0"
    env: MiniGridEnv = gym.make(
            env_id,
            tile_size=32,
            render_mode="rgb_array",
            agent_pov=False,
            agent_view_size=7,
            screen_size=512,
            max_episode_steps=128,
        )
    env.reset()
    
    # test the feature coder
    test_env_feature_coder(EmptyEnvDiscFC(), env.unwrapped)
            
    test_env_feature_coder(EmptyEnvCircFC(), env.unwrapped)