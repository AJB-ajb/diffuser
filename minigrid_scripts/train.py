#0import config.locomotion

from diffuser.models.diffusion import GaussianDiffusion
from diffuser.models import TemporalUnet
from diffuser.utils import Trainer
import diffuser.utils as utils
import diffuser.datasets.normalization as normalization

from minigrid_datasets import SequenceDataset, GoalDataset
from minigrid_renderer import MinigridRenderer
from empty_env import *
import gymnasium as gym
from gymnasium import Env

from minigrid.minigrid_env import MiniGridEnv
from minigrid_base import EnvFeatureCoderBase
from empty_env import *
from minigrid_base import Episode

import numpy as np
import torch as th
import torch.nn as nn

from pathlib import Path
import pickle

import cfg
from cfg import Cfg, Experiment

cfg = cfg.empty_env_cfg
cfg.run_name = "empty_env_1"

exp = Experiment(cfg)

device = 'cuda' if th.cuda.is_available() else 'cpu'

base_path = exp.log_dir
env = gym.make(cfg.env_id, render_mode="rgb_array")
save_folder = exp.saves_path  # trainer.savepath # also, here are render results saved
results_folder = exp.results_path # trainer.results_folder

fc = feature_coder = EmptyEnvDiscFC(env_id=cfg.env_id)
observation_dim = fc.observation_dim
action_dim = fc.action_dim

episode_file = "ppo_minigrid_2e5_new_trajectories.pkl"
with open(exp.episode_path / episode_file, "rb") as f:
    episodes = pickle.load(f)

traj_max_len = max([len(ep.observations) for ep in episodes])

def episode_iterator(episodes : list[Episode]):
    for i_episode, episode in enumerate(episodes):
        observations = episode.observations
        traj_length = len(observations)

        # use the feature encoder to encode the observations and actions
        # shape (traj_length × obs_dim)
        observation_reprs = np.array([fc.raw_obs_to_repr(obs) for obs in observations])
        action_reprs = np.array([fc.action_to_repr(action) for action in episode.actions])

        #opt: pad with the last observation in order for the model to learn that it can have shorter paths than horizon; Actions.done

        reward = episode.reward
        terminals = np.zeros(traj_length)
        if reward > 1e-5:
            terminals[traj_length-1:] = 1 # mark the last observation as terminal
            
        yield {'observations' : observation_reprs, 'actions' : action_reprs, 'terminals' : terminals}

dataset = GoalDataset(env, horizon=cfg.horizon, normalizer = normalization.LimitsNormalizer, episode_itr = episode_iterator(episodes), max_path_length = 1000, max_n_episodes = 10000, termination_penalty = None, use_padding = True)

#renderer:
renderer = MinigridRenderer(env, feature_coder)

    
# instantiate model
model = TemporalUnet(
    horizon=cfg.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=cfg.dim_mults
).to(device)

# instantiate diffusion
diffusion = GaussianDiffusion(
    model,
    horizon=cfg.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=cfg.n_diffusion_steps,
    loss_type='l1',
    clip_denoised=cfg.clip_denoised,
    predict_epsilon=cfg.predict_epsilon,
    action_weight=cfg.action_weight,
    loss_discount=1
).to(device)

trainer = Trainer(
    diffusion_model=diffusion,
    dataset=dataset,
    renderer = renderer,
    ema_decay=0.995,
    train_batch_size=32,
    train_lr=2e-4,
    gradient_accumulate_every=2,
    #step_start_ema=
    # update_ema_every=,
    log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder=str(results_folder),
        n_reference=50,
        n_samples=10,
        bucket=None,
)
trainer.n_train_steps = int(2e6)
trainer.n_steps_per_epoch=10000
trainer.savepath = str(save_folder)

# from maze2d.py base config (?)
#    loss_type='l2',
#    gradient_accumulate_every=2,
#    save_freq=1000,
#    sample_freq=1000,
#    n_saves=50,

if __name__ == '__main__':
    utils.report_parameters(model)

    print('Testing forward...', end=' ', flush=True)
    batch = utils.batchify(dataset[0])
    loss, _ = diffusion.loss(*batch)
    loss.backward()
    print('✓')

    n_epochs = int(trainer.n_train_steps // trainer.n_steps_per_epoch)

    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs} | {trainer.savepath}')
        trainer.train(n_train_steps=trainer.n_steps_per_epoch)