# run a given policy on a minigrid environment

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

def evaluate_policy(policy, env, n_episodes=10):
    episode_rewards = []
    n_successes = 0
    for _ in range(n_episodes):
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

class PolicyRunner:
    def __init__(
        self,
        env: Env,
        policy, 
        env_feature_coder: EnvFeatureCoderBase,
        seed=None,
        N_episodes2collect=100,
        render_on_screen=True,

    ) -> None:
        self.env = env
        self.base_env = env.unwrapped
        self.policy = policy
        self.env_feature_coder = env_feature_coder
        self.seed = seed
        self.closed = False
        self.render_on_screen = render_on_screen

        self.N_episodes2collect = N_episodes2collect

        self.collected_episodes = []
        self.cur_observations = []
        self.cur_actions = []

    def run(self):
        obs, info = self.reset(self.seed)

        # at random with this probability, the agent will take a random action from the action space
        # this is for more diversity in the trajectories and actions, otherwise we get a lot forward actions
        rand_probs = [0.0, 0.2, 0.4]
        cur_rand_prob = 0.0

        reward_episode = 0
        while not self.closed:
            policy = self.policy
            if np.random.rand() < cur_rand_prob:
                # take a random action of the first three actions
                action = np.random.choice(self.env_feature_coder.encoded_actions)
            else:
                action, _ = policy.predict(obs, deterministic = False)

            raw_obs = self.env_feature_coder.raw_obs_from_env(self.base_env)
            self.cur_observations.append(raw_obs)
            self.cur_actions.append(action)
            
            obs, reward, terminated, truncated, info = self.env.step(action)

            #print(f"step={self.env.unwrapped.step_count}, reward={reward:.2f}")
            reward_episode += reward

            if terminated or truncated:

                self.collected_episodes.append(Episode(observations=self.cur_observations, actions=self.cur_actions, reward=reward_episode))

                print(f"Episode: {len(self.collected_episodes)} / {self.N_episodes2collect} reward: {reward_episode}, steps: {self.env.unwrapped.step_count}")
                
                self.cur_observations, self.cur_actions = [], []
                reward_episode = 0

                # draw a new random probability
                cur_rand_prob = np.random.choice(rand_probs)
                print("New rand action probability: ", cur_rand_prob)

                if len(self.collected_episodes) >= self.N_episodes2collect:
                    self.closed = True

            if terminated:
                print("terminated!")
                self.reset(seed = None)
            elif truncated:
                print("truncated!")
                self.reset(seed = None)
            else:
                if self.render_on_screen:
                    unwrapped = self.env.unwrapped
                    unwrapped.render_mode = "human"
                    unwrapped.render()
                    unwrapped.render_mode = "rgb_array"
                    time.sleep(0.1)



    def reset(self, seed=None):
        # set random agent start position
        if seed is not None:
            np.random.seed(seed)

        unwrapped = self.env.unwrapped
        grid = unwrapped.grid
        x, y = np.random.randint([1, 1], [grid.width-1, grid.height-1])
        unwrapped.agent_start_pos = (x, y)

        ret = self.env.reset(seed=seed)
        if self.render_on_screen:
            unwrapped.render_mode = "human"
            unwrapped.render()
            unwrapped.render_mode = "rgb_array"
        return ret
    
import mgcfg

cfg = mgcfg.base_cfg
exp = mgcfg.Experiment(cfg)
# cfg.collection.id = "policy_random_1000"

env_id = cfg.env_id
tile_size = 32
agent_view = False
agent_view_size = 7
screen_size = 512
env: MiniGridEnv = gym.make(
        env_id,
        tile_size=tile_size,
        render_mode="rgb_array",
        agent_pov=agent_view,
        agent_view_size=agent_view_size,
        screen_size=screen_size,
        max_episode_steps=cfg.horizon,
    )

env = ImgObsWrapper(env)

policy_path = exp.policy_path
policy = PPO.load(policy_path, env=env)

coder = empty_env.EmptyEnvCircFC(env_id=env_id)
exec = PolicyRunner(env=env, policy=policy, env_feature_coder=coder, seed=None, N_episodes2collect=cfg.collection.n_episodes, render_on_screen=False)

exec.run()

state_consistencies = []
action_consistencies = []
for episode in exec.collected_episodes:
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

# pad trajectories with the last state and add terminal array
store = True
if store:
    import pickle
    # store trajectories and rewards in a pickle
    with open(exp.collected_episodes_path, 'wb') as f:
        pickle.dump(exec.collected_episodes, f)