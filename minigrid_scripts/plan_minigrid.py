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

import train as script
env = script.env
env.unwrapped.render_mode = 'human'

#---------------------------------- loading ----------------------------------#

#diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

dataset = script.dataset
renderer = script.renderer
script.trainer.load(epoch = 0) # epoch 0 is for all models in 0:9999 steps
diffusion = script.trainer.ema_model # / trainer.model; see what works better
obs_dim = script.observation_dim
action_dim = script.action_dim

policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#

observation, info = env.reset()

def eval_consistency(base_env, observations, actions, until_termination = True):
    """"
    Compute the ratio of actions that are consistent with the observations based on the environment.
    Note that the environment is reset at each step to the state of the observation.
    """
    initial_state = state_from_env(base_env)
    assert len(observations) == len(actions)
    n_actions = len(observations) - 1
    
    N_consistent = 0
    for i in range(n_actions): 
        state = observation_from_sampled(observations[i])
        reset_env_state(base_env, state)

        action = action_from_repr(actions[i])
        obs, reward, term, trunc, info = base_env.step(action)
        next_state = state_from_env(base_env)
        if np.all(next_state == observation_from_sampled(observations[i+1])):
            N_consistent += 1
        if term and until_termination:
            n_actions = i + 1
            break 
        
    print("N_consistent: ", N_consistent)
    ratio = N_consistent / (n_actions - 1)
    print("Consistency ratio: ", ratio)

    reset_env_state(base_env, initial_state)
    return ratio


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

    64 - 1: target_obs, # before: horizon - 1
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
