import time
from pathlib import Path

import gymnasium as gym
import pygame
from gymnasium import Env
import numpy as np

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

from stable_baselines3 import PPO

import empty_env
from minigrid_base import EnvFeatureCoderBase, Episode
import mgcfg
import tqdm


def evaluate_policy(policy, env, n_episodes=10):
    episode_rewards = []
    n_successes = 0
    for _ in tqdm.tqdm(range(n_episodes)):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done and not trunc:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            episode_reward += reward

        episode_rewards.append(episode_reward)
        if done:
            n_successes += 1


    avg_reward = np.mean(episode_rewards)
    avg_success = n_successes / n_episodes

    return {"avg_reward": avg_reward, "avg_success": avg_success}

def collect_episodes(policy, env, coder : EnvFeatureCoderBase, cfg: mgcfg.Cfg):
    """
    Collects trajectories using the policy and the environment.
    """
    rewards = []

    base_env : MiniGridEnv = env.unwrapped

    episodes = []
    n_successes = 0

    for i_traj in tqdm.tqdm(range(cfg.collection.n_episodes)):

        # reset to random location and direction
        base_env.agent_start_pos = np.random.randint([1, 1], [base_env.width-1, base_env.height-1])
        base_env.agent_start_dir = np.random.randint(0, 4)
        obs, info = env.reset()

        done = False
        episode_reward = 0

        cur_observation = [coder.obs_repr_from_env(base_env)]
        cur_actions = []

        random_prob = np.random.choice([0.0, 0.2, 0.4])
        done = trunc = False
        

        while not done and not trunc:
            if np.random.rand() < random_prob:
                action = np.random.randint(0, 3)
            else:
                action, _ = policy.predict(obs, deterministic=True)

            obs, reward, done, trunc, info = env.step(action)
            episode_reward += reward

            cur_actions.append(action)
            cur_observation.append(coder.obs_repr_from_env(base_env))

        cur_actions.append(Actions.done)

        rewards.append(episode_reward)
        if done:
            n_successes += 1
        
        episodes.append(Episode(observations=cur_observation, actions=cur_actions, reward=episode_reward))

    return episodes

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2: 
        # testing
        cfg = mgcfg.empty_env_cfg
        cfg.collection.n_episodes = 5

        print("Using default empty env configuration")
    else:
        cfg = mgcfg.Cfg.load_from_args()
    exp = mgcfg.Experiment(cfg)
    exp.instantiate()

    env = exp.env
    import train_policy
    policy = train_policy.instantiate_policy(cfg)

    episodes = collect_episodes(policy, env, exp.coder, cfg)

    coder = exp.coder
    state_consistencies = []
    action_consistencies = []
    for episode in episodes:
        # test state consistency
        raw_obs = episode.observations
        obs_reprs = [coder.raw_obs_to_repr(obs) for obs in raw_obs]
        action_reprs = [coder.action_to_repr(action) for action in episode.actions]

        state_consistency = coder.state_transition_consistency(obs_reprs, verbose=True)
        action_consistency = coder.action_consistency(obs_reprs, action_reprs, verbose=True)

        state_consistencies.append(state_consistency)
        action_consistencies.append(action_consistency)
        
    print("Average state consistency (should be 1.): ", np.mean(state_consistencies))
    print("Average action consistency (should be 1.): ", np.mean(action_consistency))

    try:
        assert np.allclose(state_consistencies, 1.0)
        assert np.allclose(action_consistencies, 1.0)
    except AssertionError as e:
        print(f"Consistency check failed: {e}")

    import pickle
    with open(exp.collected_episodes_path, 'wb') as f:
        pickle.dump(episodes, f)