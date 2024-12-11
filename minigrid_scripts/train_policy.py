import torch
import torch.nn as nn
import gymnasium as gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO

import mgcfg
from minigrid.wrappers import ImgObsWrapper, FlatObsWrapper


# convnet feature extractor ; e.g. for empty env
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    

"""
MiniGrid-Empty-Random-5x5-v0: &minigrid-defaults
  env_wrapper: minigrid.wrappers.FlatObsWrapper # See GH/1320#issuecomment-1421108191
  normalize: true
  n_envs: 8 # number of environment copies running in parallel
  n_timesteps: !!float 1e5
  policy: 'MlpPolicy'
  n_steps: 128 # batch size is n_steps * n_env
  batch_size: 64 # Number of training minibatches per update
  gae_lambda: 0.95 #  Factor for trade-off of bias vs variance for Generalized Advantage Estimator
  gamma: 0.99
  n_epochs: 10 #  Number of epoch when optimizing the surrogate
  ent_coef: 0.0 # Entropy coefficient for the loss calculation
  learning_rate: 2.5e-4 # The learning rate, it can be a function
  clip_range: 0.2
"""

def instantiate_policy(cfg):
    env = gym.make(cfg.env_id, render_mode="rgb_array")
    if cfg.policy.name == 'CnnPolicy':
        env = ImgObsWrapper(env)
        policy_kwargs = dict(features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128))

        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    elif cfg.policy.name == 'MlpPolicy':
        env = FlatObsWrapper(env)
        model = PPO(cfg.policy, env, verbose=1)
        
    else:
        raise NotImplementedError(f"Policy {cfg.policy.name}not implemented")

    return model
    

if __name__ == '__main__':
    cfg = mgcfg.Cfg.load_from_args()
    exp = mgcfg.Experiment(cfg)
    
    env = gym.make(cfg.env_id, render_mode="rgb_array")

    model = instantiate_policy(cfg)

    model.learn(total_timesteps=int(cfg.policy.n_timesteps))
    model.save(exp.policy_path)


