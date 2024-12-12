"""
    Infrastructure for managing configuration and results.
    
    For each environment, we might collect multiple sets of trajectories or policies, which are stored in their respective folders.

    The experiment captures the runtime state.
"""

import json
import collections.abc
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# minigrid-diffuser experiment configuration
class Cfg(collections.abc.MutableMapping):
    def __init__(self, is_toplevel = False, **kwargs):

        if is_toplevel:
            self.storage_dir = Path(__file__).parents[1] / "data"
            self.is_toplevel = is_toplevel
        
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.__dict__[key] = Cfg(**value)
            else:
                self.__dict__[key] = value

    def __setitem__(self, name, value):
        if name in self.__dict__.keys():
            self.__dict__[name] = value
        else:
            raise AttributeError(f"Attribute {name} unavailable. For setting a new attribute, use the . attribute syntax. ")
    def __getitem__(self, name):
        return self.__dict__[name]
    def __delitem__(self, key):
        return self.__dict__.__delitem__(key)
    def __iter__(self):
        return iter(self.__dict__)
    def __len__(self):
        return len(self.__dict__)
    
    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, Cfg):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)
    
    def id(self):
        """
            Return an identifier for the values of the configuration.
            Concatenates the displayable values of the configuration, floats in the format .2e .
        """
        def value(v):
            if isinstance(v, Cfg):
                return v.id()
            elif isinstance(v, float):
                return f"{v:.2e}"
            else:
                return str(v)
        return "_".join([value(v) for v in self.__dict__.values()])
    
    @staticmethod
    def load_from_json(path): 
        """
            Load configuration from json file.
            Assumes the config is in storage_dir / cfgs / name.json
            and infers the storage dir from the location.
        """
        with open(str(path), "r") as f:
            cfg_dict = json.load(f)
        
        cfg = Cfg(**cfg_dict)
        cfg.storage_dir = Path(path).parents[1]
        return cfg
    
    @staticmethod
    def load_from_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default=None)
        args = parser.parse_args()
        cfg_path = args.cfg
        cfg = Cfg.load_from_json(cfg_path)
        return cfg
    
    def vars(self):
        """
            Return all variables (not paths) in a dictionary
        """
        def value(v):
            if isinstance(v, Cfg):
                return v.vars()
            else:
                return v
        return {k: value(v) for k, v in self.__dict__.items() if not isinstance(v, Path)}
    
    @property
    def cfg_path(self):
        """
            Return the path where the configuration is stored.
        """
        return self.storage_dir / "cfgs" / f"{self.name}.json"
    
    def save_to_json(self, path = None):
        """
            Save configuration to json file.
            If path is none, saves to storage_dir / cfgs / name.json
            """
        if path is None:
            path = self.cfg_path
            path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(str(path), "w") as f:
            json.dump(self.vars(), f)

    
        
class Experiment:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.log_dir = cfg.storage_dir / cfg.name 
        self.run_log_dir = self.log_dir / cfg.run_name

        self.tensorboard_logdir = self.run_log_dir / "logs"

        for dir in [self.log_dir, self.tensorboard_logdir, self.model_dir, self.results_dir, self.saves_dir, self.episode_dir]:
            dir.mkdir(exist_ok=True, parents=True)

    
    def start_logging(self):
        self.tensorboard_logdir.mkdir(exist_ok=True, parents=True)
        self.summary_writer = SummaryWriter(log_dir=self.tensorboard_logdir)

    def instantiate(self):
        """
            Instantiate the environment and the feature coder    
        """
        import gymnasium as gym
        from minigrid.wrappers import ImgObsWrapper

        env = gym.make(self.cfg.env_id, render_mode="rgb_array", max_episode_steps=self.cfg.horizon-2) # -1 steps needed to have at max horizon observations; somehow -2 needed to not have some dataset error
        self.env = ImgObsWrapper(env)

        import importlib
        self.env_module = importlib.import_module(self.cfg.env_module)
        coder_class = getattr(self.env_module, self.cfg.feature_coder)
        
        self.coder = coder_class(env_id=self.cfg.env_id)
        self.summary_writer = SummaryWriter(log_dir=self.tensorboard_logdir)

    @property 
    def model_dir(self): # to save trained diffuser models
        return self.run_log_dir / "model"
    
    # results plots
    @property
    def results_dir(self):
        return self.run_log_dir / "results"
    
    @property
    def saves_dir(self):
        return self.run_log_dir / "saves"
    
    @property
    def repo_dir(self):
        return Path(__file__).parents[1]
    
    @property
    def policy_path(self): # standard path to load the policy from
        return self.log_dir / "agent_models" / f"{self.cfg.policy.id}.zip"
    
    # where to store the collected episodes
    
    @property
    def episode_dir(self):
        return self.log_dir / "episodes"
    @property
    def collected_episodes_path(self):
        return self.episode_dir / f"{self.cfg.collection.id}.pkl"
    

# base configuration for the empty environment
base_cfg = Cfg(
    is_toplevel=True,
    name = "",
    env_id="",
    run_name="",
    env_module = "",
    feature_coder = "",
    policy=Cfg(
        id="cnn_2e5",
        name='CnnPolicy',
        features_extractor_class='MinigridFeaturesExtractor',
        features_dim=128,
        policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=128)),
        n_timesteps=2e5
    ),
    collection = Cfg(
        id="policy_random_1000", # default: use policy and random exploration in between
        n_episodes=1000,
        exploration_probs=[0.0, 0.2, 0.4]
    ),
    horizon=128,
    max_path_length=128,
    dim_mults=[1, 4, 8],
    diffusion=Cfg(
        n_diffusion_steps=64,
        action_weight=1,
        loss_discount=1,
        predict_epsilon=False,
        clip_denoised=True,
        use_padding=False
    ),
    trainer=Cfg(
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-4,
        gradient_accumulate_every=2,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        n_reference=50,
        n_samples=10,
        bucket=None,
        n_train_steps=int(2e6),
        n_steps_per_epoch=10000
    )
)

empty_env_cfg = Cfg(**base_cfg)
empty_env_cfg['name'] = "empty_env"
empty_env_cfg['run_name'] = "run0"
empty_env_cfg['env_id'] = "MiniGrid-Empty-16x16-v0"
empty_env_cfg['env_module'] = "empty_env"
empty_env_cfg['feature_coder'] = "EmptyEnvDiscFC"
empty_env_cfg['policy']['id'] = "cnn_2e5"
empty_env_cfg['policy']['n_timesteps'] = 2e5
empty_env_cfg['collection']['id'] = "policy_random_1000"
empty_env_cfg['collection']['n_episodes'] = 1000


if __name__ == "__main__":
    # test loading and saving
    cfg = base_cfg
    exp = Experiment(cfg)

    cfg.save_to_json()
    cfg2 = Cfg.load_from_json(cfg.storage_dir / "cfgs" / f"{cfg.name}.json")

    print(cfg)
    print(cfg2)

    for (k1, v1), (k2, v2) in zip(cfg.vars().items(), cfg2.vars().items()):
        assert k1 == k2
        assert v1 == v2

    assert cfg == cfg2