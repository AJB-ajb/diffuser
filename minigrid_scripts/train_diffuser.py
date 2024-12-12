#0import config.locomotion
import math

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
from minigrid_base import Episode

import numpy as np
import torch as th
import torch.nn as nn

from pathlib import Path
import pickle

import mgcfg
from mgcfg import Cfg, Experiment
import sys
from typing import List

if len(sys.argv) > 1:
    cfg = mgcfg.Cfg.load_from_args()
else:
    # test
    cfg = mgcfg.empty_env_cfg
    cfg['horizon'] = 32
    cfg.trainer['n_train_steps'] = 1000
    cfg['max_path_length'] = 2*cfg.horizon # must be larger than horizon
    cfg['feature_coder'] = 'EmptyEnvDiscFC'
    print("Using default empty env configuration for testing")


exp = Experiment(cfg)
exp.instantiate()

device = 'cuda' if th.cuda.is_available() else 'cpu'

env = gym.make(cfg.env_id, render_mode="rgb_array")

fc = feature_coder = exp.coder
observation_dim = fc.observation_dim
action_dim = fc.action_dim

with open(str(exp.collected_episodes_path), "rb") as f:
    episodes = pickle.load(f)

traj_max_len = max([len(ep.observations) for ep in episodes])

def episode_iterator(episodes: List[Episode]):
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

dataset = GoalDataset(
    env,
    horizon=cfg.horizon,
    normalizer=normalization.LimitsNormalizer,
    episode_itr=episode_iterator(episodes),
    max_path_length=cfg.max_path_length, 
    max_n_episodes=10000,
    termination_penalty=None,
    use_padding=True
)

#renderer:
renderer = MinigridRenderer(exp, cfg)

    
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
    n_timesteps=cfg.diffusion.n_diffusion_steps,
    loss_type='l1',
    clip_denoised=cfg.diffusion.clip_denoised,
    predict_epsilon=cfg.diffusion.predict_epsilon,
    action_weight=cfg.diffusion.action_weight,
    loss_discount=cfg.diffusion.loss_discount
).to(device)


try:
    print("Compiling model and diffusion...")
    # if compile is available, use it
    #torch.set_float32_matmul_precision('high') 
    model = th.compile(model) # improves train speed by roughly 2% - 10%
    model = th.compile(diffusion)

    print("Model compiled successfully")

except Exception as e:
    print(f"Error while compiling model and diffusion: {e}")


trainer = Trainer(
    diffusion_model=diffusion,
    dataset=dataset,
    renderer=renderer,
    ema_decay=cfg.trainer.ema_decay,
    results_folder=str(exp.results_dir),
    train_batch_size=cfg.trainer.train_batch_size,
    train_lr=cfg.trainer.train_lr,
    gradient_accumulate_every=cfg.trainer.gradient_accumulate_every,
    log_freq=cfg.trainer.log_freq,
    sample_freq=cfg.trainer.sample_freq,
    save_freq=cfg.trainer.save_freq,
    label_freq=cfg.trainer.label_freq,
    save_parallel=cfg.trainer.save_parallel,
    n_reference=cfg.trainer.n_reference,
    n_samples=cfg.trainer.n_samples,
    bucket=cfg.trainer.bucket,
)
trainer.n_train_steps = cfg.trainer.n_train_steps
trainer.n_steps_per_epoch = cfg.trainer.n_steps_per_epoch
trainer.savepath = str(exp.saves_dir)

exp.trainer = trainer
trainer.cfg = cfg

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

    n_epochs = math.ceil(trainer.n_train_steps / trainer.n_steps_per_epoch)

    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs} | {trainer.savepath}')
        trainer.train(n_train_steps=max(trainer.n_steps_per_epoch, trainer.n_train_steps - trainer.step))