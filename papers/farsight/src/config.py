import os
import yaml
import math


class Config:
    """
    Configuration loader for FarSight project.
    Loads default and optional eval configuration YAML files,
    merges them, and computes additional parameters (e.g., sigma).
    """

    def __init__(self, default_path=None, eval_path=None):
        # Determine project root directory (one level up from src/)
        root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir)
        )
        # Default config file path
        if default_path is None:
            default_path = os.path.join(root_dir, "configs", "default.yaml")
        # Load default config
        if not os.path.exists(default_path):
            raise FileNotFoundError(f"Default config file not found: {default_path}")
        with open(default_path, 'r') as f:
            default_cfg = yaml.safe_load(f)

        print
        # Load eval config if provided
        eval_cfg = {}
        if eval_path:
            if not os.path.exists(eval_path):
                raise FileNotFoundError(f"Eval config file not found: {eval_path}")
            with open(eval_path, 'r') as f:
                eval_cfg = yaml.safe_load(f)

        print(f"Loaded default config from: {default_path}")
    
        # Merge configs: eval overrides default
        self.cfg = default_cfg.copy()
        for key, value in eval_cfg.items():
            if key in self.cfg and isinstance(value, dict):
                for subkey, subvalue in value.items():
                    self.cfg[key][subkey] = subvalue
            else:
                self.cfg[key] = value

        # Expose common sections
        self.model = self.cfg.get('model', {})
        self.data = self.cfg.get('data', {})
        self.hyperparameters = self.cfg.get('hyperparameters', {})
        self.logging = self.cfg.get('logging', {})

        # Compute sigma for causal attention register
        decay_base = self.hyperparameters.get('decay_base')
        seq_len = self.hyperparameters.get('seq_len')
        if decay_base is None or seq_len is None:
            raise KeyError("Both 'decay_base' and 'seq_len' must be specified in hyperparameters.")
        sigma = math.log(decay_base) / seq_len
        self.hyperparameters['sigma'] = sigma

    def __getitem__(self, key):
        return self.cfg[key]

    def get(self, key, default=None):
        return self.cfg.get(key, default)
