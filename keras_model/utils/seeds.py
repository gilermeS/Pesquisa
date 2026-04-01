"""Reproducibility utilities for TensorFlow and related libraries."""
import os
import random
import numpy as np
import tensorflow as tf
from typing import Optional


def set_seed(
    seed: int = 42,
    enable_tf_determinism: bool = True,
    warn_only: bool = False
) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed to set
        enable_tf_determinism: Enable TF_DETERMINISTIC_OPS (may impact performance)
        warn_only: If True, don't enforce determinism and just warn
        
    Note:
        Full determinism in TensorFlow requires:
        1. Setting all random seeds
        2. Setting TF_DETERMINISTIC_OPS=1
        3. Using single thread for operations (may slow training)
        
        Some operations remain non-deterministic even with these settings
        (e.g., cuDNN convolution on certain GPUs).
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1' if enable_tf_determinism else '0'
    
    if enable_tf_determinism and not warn_only:
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def get_seed() -> Optional[int]:
    """Get the current random seed (from environment)."""
    seed_str = os.environ.get('PYTHONHASHSEED', None)
    return int(seed_str) if seed_str else None


class SeedContext:
    """Context manager for temporary seed setting."""
    
    def __init__(self, seed: int, **kwargs):
        self.seed = seed
        self.kwargs = kwargs
        self.previous_seeds = {}
    
    def __enter__(self):
        self.previous_seeds['PYTHONHASHSEED'] = os.environ.get('PYTHONHASHSEED')
        self.previous_seeds['TF_DETERMINISTIC_OPS'] = os.environ.get('TF_DETERMINISTIC_OPS')
        set_seed(self.seed, **self.kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.previous_seeds.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)