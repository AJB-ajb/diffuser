# general functionality for minigrid
from dataclasses import dataclass
import numpy as np
from numpy.linalg import norm


from abc import ABC, abstractmethod

@dataclass
class Episode:
    observations: list
    actions: list
    reward: float

# abstract class that defines how the features of any environment are encoded and decoded
# we are defining different encodings to see how well these are learned by diffuser, and also for different MinigridEnvironments; e.g. the UnlockPickup task needs a larger state space which includes the key and door positions

class EnvFeatureCoderBase(ABC):
    """
    Abstract class that defines how the features of any environment are encoded and decoded.
    """

    # ------------------- abstract methods ------------------- #
    # dependent on environment and needed actions:
    @abstractmethod
    def raw_obs_from_env(self, base_env):
        """
        Return the current state of the environment as numpy array in its raw (unencoded) version.
        """

    @abstractmethod
    def reset_env_state(self, base_env, raw_obs):
        """
        Reset the environment to the state encoded by `raw_obs`.
        Note that `raw_obs` must contain only discrete values (it is the non-rounded version).
        """

    # ------------------- dependent on specific encoding ------------------- #
    @abstractmethod
    def raw_obs_to_repr(self, raw_obs):
        """
            Return the (encoded) observation representation from the raw observation.
        """

    @abstractmethod
    def action_to_repr(self, action):
        """
        Return the encoded action representation of the discrete minigrid action.
        """

    @abstractmethod
    def action_from_repr(self, action_repr):
        """ 
        Return the approximate minigrid action from the encoded continuous action representation.
        """
    
    @abstractmethod
    def raw_obs_from_repr(self, obs_repr):
        """
        Return the approximate observation representation from the encoded observation representation. 
        For example, the continuous observation representation could be rounded to the nearest integer.
        """
    
    @abstractmethod
    def calc_state_transition(self, obs_repr1, obs_repr2):
        """
            Calculate the most likely action as well as absolute position and direction change between the two states.
            Return: action, (norm_Δpos, abs_Δdir, …)
        """

    def state_transition_consistency(self, obs_reprs, ε = 1e-3, verbose = True):
        """
            Return (num_transitions_consistent / num_transitions) for the given sequence of observations. 
            An observation is considered to be consistent if there is a legal action that transitions the agent to the next observation.
        """
        self.reset_env_state(self._base_env, self.raw_obs_from_repr(obs_reprs[0]))
        N_consistent = 0

        for i in range(len(obs_reprs) - 1):
            obs1 = obs_reprs[i]
            obs2 = obs_reprs[i + 1]
            action, Δs = self.calc_state_transition(obs1, obs2)
            self._base_env.step(action)

            next_obs_repr = self.obs_repr_from_env(self._base_env)
            if norm(next_obs_repr - obs2) < ε:
                N_consistent += 1
            else: 
                if verbose:
                    print(f"Transition {i} is inconsistent: {obs1} -> {obs2}")
                    print(f"Expected: {obs2}, got: {next_obs_repr}")
                    print(f"Action: {action}, Δs: {Δs}")

            self.reset_env_state(self._base_env, self.raw_obs_from_repr(obs2))

        return N_consistent / (len(obs_reprs) - 1)
    
    def action_consistency(self, obs_reprs, action_reprs, ε = 1e-3, verbose = True):
        """
            Return (num_actions_consistent / num_actions) for the given sequence of observations and actions.
            An action is considered to be consistent if it transitions the agent sufficiently close to the next expected observation.
        """
        self.reset_env_state(self._base_env, self.raw_obs_from_repr(obs_reprs[0]))
        N_consistent = 0

        for i in range(len(obs_reprs) - 1):
            obs1 = obs_reprs[i]
            obs2 = obs_reprs[i + 1]
            action = self.action_from_repr(action_reprs[i])
            self._base_env.step(action)

            next_obs_repr = self.obs_repr_from_env(self._base_env)
            if norm(next_obs_repr - obs2) < ε:
                N_consistent += 1
            else: 
                if verbose:
                    print(f"Action {i} is inconsistent: {obs1} -> {obs2}")
                    print(f"Expected: {obs2}, got: {next_obs_repr}")
                    print(f"Action: {action}")

            self.reset_env_state(self._base_env, self.raw_obs_from_repr(obs2))

        return N_consistent / (len(action_reprs) - 1)
    


    def obs_repr_from_env(self, base_env):
        """
        Return the current state of the environment as numpy array, as encoded by this encoder.
        """
        return self.raw_obs_to_repr(self.raw_obs_from_env(base_env))


    def get_transition(self, base_env, discrete_action):
        """
        Return the current (observation_representation, action_representation) as numpy array.
        """
        obs_repr = self.obs_repr_from_env(base_env)
        action_repr = self.action_to_repr(discrete_action)
        return np.concatenate([obs_repr, action_repr])
    
    def repr_diff(self, sampled_observation):
        """
        Return the numerical difference between the sampled state and the encoded state that is inferred from the sampled state. 
        """
        raw_obs = self.raw_obs_from_repr(sampled_observation)
        obs_repr_discretized = self.raw_obs_to_repr(raw_obs)
        return np.linalg.norm(obs_repr_discretized - sampled_observation, ord = 1)