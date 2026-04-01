"""Custom callbacks for training."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import callbacks as keras_callbacks
from pathlib import Path
from typing import Optional, List, Dict, Any
import json


class MetricLogger(keras_callbacks.Callback):
    """Log metrics to JSON file during training."""
    
    def __init__(self, log_dir: str, filename: str = 'metrics.json'):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self.history = {}
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(float(value))
        
        with open(self.filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


class LearningRateSchedulerWithWarmup(keras_callbacks.Callback):
    """Learning rate scheduler with warmup period."""
    
    def __init__(
        self,
        initial_lr: float = 1e-3,
        warmup_epochs: int = 5,
        decay_type: str = 'exponential',
        decay_rate: float = 0.5,
        decay_steps: int = 10,
        min_lr: float = 1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            if self.decay_type == 'exponential':
                lr = self.initial_lr * (self.decay_rate ** ((epoch - self.warmup_epochs) / self.decay_steps))
            elif self.decay_type == 'cosine':
                lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * (epoch - self.warmup_epochs) / (self.decay_steps)))
            else:
                lr = self.initial_lr
        
        lr = max(lr, self.min_lr)
        keras.backend.set_value(self.model.optimizer.lr, lr)


def create_standard_callbacks(
    save_dir: Optional[Path] = None,
    monitor: str = 'val_loss',
    patience: int = 10,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    log_metrics: bool = True,
) -> List[keras.callbacks.Callback]:
    """Create a standard set of training callbacks."""
    callbacks = []
    
    callbacks.append(keras_callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=1e-4,
        patience=patience,
        restore_best_weights=True,
        verbose=1
    ))
    
    callbacks.append(keras_callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        verbose=1
    ))
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks.append(keras_callbacks.ModelCheckpoint(
            filepath=str(save_dir / 'best_model.keras'),
            monitor=monitor,
            save_best_only=True,
            restore_best_weights=True,
            verbose=0
        ))
        
        callbacks.append(keras_callbacks.CSVLogger(
            str(save_dir / 'training_log.csv'),
            separator=','
        ))
        
        if log_metrics:
            callbacks.append(MetricLogger(str(save_dir)))
    
    return callbacks