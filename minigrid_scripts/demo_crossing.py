import minigrid
from minigrid.wrappers import FullyObsWrapper
import gymnasium as gym
import time

#env = gym.make("MiniGrid-Empty-5x5-v0")
#env.reset(seed=0)

# Create a MiniGrid environment
env = gym.make("MiniGrid-Empty-8x8-v0")  # Replace with desired environment ID

# Optionally wrap the environment for a fully observable grid
env = FullyObsWrapper(env)  # Provides a full grid observation

# Reset the environment to start
obs, info = env.reset()

# Render the environment
for i in range(100):
    action = env.action_space.sample()  # Sample random action
    obs, reward, done, _, info = env.step(action)  # Take a step
    env.render()  # By default, this shows a graphical window
    # sleep
    time.sleep(0.5)

# Close the environment when done
env.close()

