"""
    Infrastructure for managing configuration and results.
    
    For each environment, we might collect multiple sets of trajectories or policies, which are stored in their respective folders.

    The experiment captures the runtime state.
"""

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# minigrid-diffuser experiment configuration
class Cfg:
    def __init__(self, **kwargs):

        self.storage_path = Path(__file__).parents[1] / "data"

        self.__dict__.update(kwargs)
        # if on colab, set to google drive folder

    def __setitem__(self, name, value):
        if name in self.__dict__.keys():
            self.__dict__[name] = value
        else:
            raise AttributeError(f"Attribute {name} unavailable. For setting a new attribute, use the . attribute syntax. ")
    def __getitem__(self, name):
        return self.__dict__[name]
        
class Experiment:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.log_dir = cfg.storage_path / cfg.run_name
        self.tensorboard_logdir = self.log_dir / "logs"
    
    def start_logging(self):
        self.tensorboard_logdir.mkdir(exist_ok=True, parents=True)
        self.summary_writer = SummaryWriter(log_dir=self.tensorboard_logdir)

    @property 
    def model_path(self):
        return self.log_dir / "model"
    
    @property
    def results_path(self):
        return self.cfg.storage_path / self.cfg.env_id
    
    @property
    def saves_path(self):
        return self.log_dir / "saves"
    
    @property
    def repo_path(self):
        return Path(__file__).parents[1]
    
    @property
    def policy_path(self): # standard path to load the policy from
        # td use glob and the most recent
        return self.results_path / "agent_models" / "ppo.zip"
    
    @property
    def episode_path(self):
        return self.results_path / "episodes"
    
    @property
    def trajectory_path(self):
        return self.results_path / "trajectories"


# base configuration for the empty environment
empty_env_cfg = Cfg(env_id = "MiniGrid-Empty-16x16-v0",
                      horizon = 128,
                      
                      n_diffusion_steps = 64,
                      action_weight = 1,
                      loss_discount = 1,
                      predict_epsilon = False,
                      dim_mults = (1, 4, 8),
                      clip_denoised = True,
                      use_padding = False
                     )

