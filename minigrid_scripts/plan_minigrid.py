import json
import numpy as np
from os.path import join
import time

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import minigrid
from minigrid.core.actions import Actions

from empty_env import *


#---------------------------------- setup ----------------------------------#

# todo use config
import sys
import mgcfg

import train_diffuser as script

if len(sys.argv) > 1:
    cfg = mgcfg.Cfg.load_from_args()
else:
    # test
    cfg = mgcfg.empty_env_cfg
    print("Using default empty env configuration for testing")

exp = script.Experiment(cfg)

device = 'cuda' if th.cuda.is_available() else 'cpu'

env = gym.make(cfg.env_id, render_mode="human")

fc = feature_coder = script.EmptyEnvDiscFC(env_id=cfg.env_id)
observation_dim = fc.observation_dim
action_dim = fc.action_dim

episode_file = "ppo_minigrid_2e5_new_trajectories.pkl"
with open(exp.episode_dir / episode_file, "rb") as f:
    episodes = pickle.load(f)

traj_max_len = max([len(ep.observations) for ep in episodes])

def episode_iterator(episodes : list[script.Episode]):
    # ...existing code...

dataset = GoalDataset(env, horizon=cfg.horizon, normalizer=script.normalization.LimitsNormalizer, episode_itr=episode_iterator(episodes), max_path_length=1000, max_n_episodes=10000, termination_penalty=None, use_padding=True)

renderer = MinigridRenderer(env, feature_coder)

model = TemporalUnet(
    horizon=cfg.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=cfg.dim_mults
).to(device)

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

#---------------------------------- loading ----------------------------------#

trainer.load(epoch=0)
diffusion = trainer.ema_model
obs_dim = script.observation_dim
action_dim = script.action_dim

policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#

observation, info = env.reset()

# find goal position in grid

def find_goal(grid):
    for x in range(1, grid.width-1):
        for y in range(1, grid.height-1):
            if isinstance(grid.get(x, y), minigrid.core.world_object.Goal):
                target = np.array([x, y])
                return target
                break
    return None

    
base_env = env.unwrapped
grid = base_env.grid
target = find_goal(grid)

## set conditioning xy position to be the goal
target_obs = np.zeros(obs_dim)
target_obs[:2] = target

cond = {

    cfg.horizon - 1: target_obs,
}

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
for t in range(base_env.max_steps):

    state = state_from_env(base_env)

    # plan once, then follow the plan, i.e. open-loop control
    if t == 0:
        cond[0] = state

        action, samples = policy(cond, batch_size=1)
        actions = samples.actions[0]
        sequence = samples.observations[0]

        eval_consistency(base_env, sequence, actions)

        print("Actions: ", actions)
        print("Observations: ", sequence)

    # ####
    if t < len(sequence) - 1:
        next_waypoint = sequence[t+1]
    else:
        next_waypoint = sequence[-1].copy()
        next_waypoint[2:] = 0

    ## can use actions or define a simple controller based on state predictions
    # action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

    action = action_from_repr(actions[t])


    next_observation, reward, terminal, trunc, _ = env.step(action)
    total_reward += reward

    #score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {-1:.4f} | '
        f'{action}'
    )
    env.render()
    time.sleep(0.2)

    ## update rollout observations
    rollout.append(next_observation.copy())

    # logger.log(score=score, step=t)

    fullpath = script.save_folder / f'{t}.png'
    if t == 0: renderer.composite(str(fullpath), samples.observations, ncol=1)

        ## save rollout thus far
        #renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

    if terminal:
        break

    observation = next_observation

## save result as a json file
#json_path = join(args.savepath, 'rollout.json')
#json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
#    'epoch_diffusion': diffusion_experiment.epoch}
#json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
