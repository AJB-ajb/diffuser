import matplotlib.pyplot as plt
import numpy as np

from diffuser.utils.rendering import plot2img, zipkw
import einops
import imageio

from minigrid.core.actions import Actions
from minigrid_base import EnvFeatureCoderBase

MAZE_BOUNDS = {
    'maze2d-umaze-v1': (0, 5, 0, 5),
    'maze2d-medium-v1': (0, 8, 0, 8),
    'maze2d-large-v1': (0, 9, 0, 12)
}

class MinigridRenderer:

    def __init__(self, env, feature_coder:EnvFeatureCoderBase ): # env should contain the base environment
        self.feature_coder = feature_coder
        self.base_env = env.unwrapped
        self.env = env
        self.grid = self.base_env.grid
        self._background = np.zeros((self.grid.width, self.grid.height)) # boolean image of gridsize, where 1 is wall and 0 is empty
        self._remove_margins = False
        self._extent = (0, 1, 1, 0) # extent of the plot (x0, x1, y0, y1)

    def renders(self, observations, conditions=None, title=None):
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.imshow(self._background * .5,
            extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)
        
        # evaluate consistency
        obs_reprs = observations
        state_trans_cons = self.feature_coder.state_transition_consistency(obs_reprs, ε = 2e-1, verbose = False)
        print(f"State transition consistency: {state_trans_cons}")

        path_length = len(observations)
        # flip direction of y 
        plt.gca().invert_yaxis()
        colors = plt.cm.jet(np.linspace(0,1,path_length))

        # plot start and end points
        plt.scatter(observations[[0, -1],0], observations[[0,-1],1], c='black', zorder=10, marker='x')
        plt.scatter(observations[:,0], observations[:,1], c=colors, zorder=20)
        # td: plot arrows in the direction of the orientation
        for i, (x, y, direction) in enumerate(observations):
            angle = direction * np.pi / 2
            head_length = 0.3
            plt.arrow(x, y, head_length * np.cos(angle), head_length * np.sin(angle), color=colors[i], head_width=0.2, head_length=0.1)
            
        # color different actions in different colors
        # where get the actions from?
        # td: colors = {Actions.forward: 'blue', Actions.left: 'green', Actions.right: 'red'}
          

        # td: plot different colors for different actions
        plt.axis('off')
        plt.title(title)
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img

    def composite(self, savepath, paths, ncol=5, **kwargs):
        '''
            savepath : str
            observations : [ n_paths x horizon x 2 ]
        '''
        assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, **kw)
            images.append(img)
        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(images,
            '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        imageio.imsave(savepath, images)
        print(f'Saved {len(paths)} samples to: {savepath}')

# todo (opt)
class Maze2dRenderer(MinigridRenderer):

    def __init__(self, env, observation_dim=None):
        self.env = env
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        self.goal = None
        self._background = self.env.maze_arr == 10
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, conditions=None, **kwargs):
        bounds = MAZE_BOUNDS[self.env_name]

        observations = observations + .5
        if len(bounds) == 2:
            _, scale = bounds
            observations /= scale
        elif len(bounds) == 4:
            _, iscale, _, jscale = bounds
            observations[:, 0] /= iscale
            observations[:, 1] /= jscale
        else:
            raise RuntimeError(f'Unrecognized bounds for {self.env_name}: {bounds}')

        if conditions is not None:
            conditions /= scale
        return super().renders(observations, conditions, **kwargs)