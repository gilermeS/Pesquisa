# KerasModel TODO - Deep Learning Infrastructure and Model Library

This document outlines a comprehensive plan for building a modular Keras model library with proper training infrastructure, tf.data pipelines, and reproducible experiment management for cosmological data analysis.

**Last Updated**: March 2026  
**Status**: Planned (not yet implemented)  
**Priority**: High  
**Estimated Effort**: 6-8 weeks

---

## Table of Contents

1. [Overview](#1-overview)
2. [Library Structure](#2-library-structure)
3. [Data Pipeline (tf.data)](#3-data-pipeline-tfdata)
4. [Model Architectures](#4-model-architectures)
5. [Training Infrastructure](#5-training-infrastructure)
6. [Experiment Tracking & Reproducibility](#6-experiment-tracking--reproducibility)
7. [Evaluation & Metrics](#7-evaluation--metrics)
8. [Hyperparameter Management](#8-hyperparameter-management)
9. [Implementation Order](#9-implementation-order)
10. [Code Examples](#10-code-examples)
11. [Backward Compatibility](#11-backward-compatibility)

---

## 1. Overview

### Current State Analysis

**Existing Models**:
- CNN: Conv1D layers with MaxPooling1D, Dense readout
- Dense: Simple fully-connected layers
- RNN: GRU layers (standard and Bidirectional)
- BNN: Bayesian Neural Network (neurobayes)
- SVM: Classical ML baseline

**Issues Identified**:

| Issue | Impact | Frequency |
|-------|--------|-----------|
| No reproducibility (random seeds) | Non-reproducible results | All models |
| Sequential numpy loading | Slow training, poor GPU utilization | All models |
| Hardcoded hyperparameters | No experimentation, manual tracking | All models |
| No experiment versioning | Difficult to reproduce specific runs | All models |
| No unified evaluation | Different metrics per notebook | All models |
| Inconsistent callbacks | Suboptimal training | All models |
| No model registry | Scattered saved models | All models |
| No hyperparameter search | Suboptimal architectures | All models |
| Missing uncertainty quantification | No confidence estimates | DL models |
| No regularization standardization | Potential overfitting | Some models |

### Goals

1. **Reproducibility**: Deterministic training with seed management
2. **Efficiency**: tf.data pipelines for GPU utilization
3. **Modularity**: Reusable model architectures and training loops
4. **Trackability**: Experiment tracking and model versioning
5. **Flexibility**: Easy hyperparameter search and architecture experimentation
6. **Evaluation**: Comprehensive metrics and uncertainty quantification

---

## 2. Library Structure

### 2.1 Directory Structure

```
keras_model/
├── __init__.py
├── _version.py
├── data/
│   ├── __init__.py
│   ├── pipelines.py      # tf.data pipeline builders
│   ├── generators.py     # Data generators
│   └── augmentation.py   # Data augmentation
├── models/
│   ├── __init__.py
│   ├── architectures.py   # CNN, Dense, RNN, etc.
│   ├── blocks.py         # Reusable building blocks
│   └── custom.py         # Custom model components
├── training/
│   ├── __init__.py
│   ├── train.py          # Training loops
│   ├── callbacks.py       # Custom callbacks
│   ├── losses.py         # Custom losses
│   └── optimizers.py     # Optimizer utilities
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py        # Custom metrics
│   ├── uncertainty.py    # Uncertainty quantification
│   └── comparison.py     # Model comparison tools
├── experiments/
│   ├── __init__.py
│   ├── tracker.py        # Experiment tracking
│   ├── registry.py       # Model registry
│   └── search.py        # Hyperparameter search
└── utils/
    ├── __init__.py
    ├── seeds.py          # Seed management
    ├── logging.py        # Logging utilities
    └── serialization.py  # Model saving/loading
```

### 2.2 Package Initialization

```python
"""KerasModel - Deep Learning Infrastructure for Cosmological Data Analysis."""

from keras_model.data import (
    create_training_pipeline,
    create_validation_pipeline,
    create_tf_dataset,
)
from keras_model.models import (
    build_cnn_model,
    build_dense_model,
    build_rnn_model,
    build_gru_model,
    build_bidirectional_rnn_model,
)
from keras_model.training import (
    train_model,
    train_with_cross_validation,
)
from keras_model.evaluation import (
    evaluate_model,
    calculate_uncertainty,
    compare_models,
)
from keras_model.experiments import (
    ExperimentTracker,
    ModelRegistry,
    HyperparameterSearch,
)

__version__ = "1.0.0"
__author__ = "Guilherme de Souza Ramos Cardoso"
```

---

## 3. Data Pipeline (tf.data)

### 3.1 Core Pipeline Builder

**File**: `keras_model/data/pipelines.py`

```python
"""tf.data pipeline builders for efficient data loading."""
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union, Callable


class CosmologicalDataPipeline:
    """
    Efficient tf.data pipeline for cosmological H(z) data.
    
    Features:
    - Parallel data loading
    - Automatic batching
    - Prefetching for GPU utilization
    - Caching for faster subsequent epochs
    - Optional shuffling
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        file_pattern: str = "data_{}.npy",
        batch_size: int = 32,
        validation_split: float = 0.2,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000,
        cache: bool = True,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        prefetch_buffer_size: int = tf.data.AUTOTUNE,
        seed: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.file_pattern = file_pattern
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.cache = cache
        self.num_parallel_calls = num_parallel_calls
        self.prefetch_buffer_size = prefetch_buffer_size
        self.seed = seed
        
        self.file_paths = sorted(self.data_dir.glob(file_pattern.replace("{}","*")))
        
        if not self.file_paths:
            raise FileNotFoundError(f"No files found in {data_dir}")
    
    def _parse_single_file(self, file_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        def parse(path):
            data = np.load(path.numpy())
            features = data[:, :2].astype(np.float32)
            n_points = len(data)
            target = data[0, 2] / n_points
            target = np.array(target, dtype=np.float32)
            return features, target
        
        features, target = tf.py_function(
            func=parse,
            inp=[file_path],
            Tout=[tf.float32, tf.float32]
        )
        
        n_points = len(np.load(self.file_paths[0]))
        features.set_shape([n_points, 2])
        target.set_shape([])
        
        return features, target
    
    def _preprocess_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        if self.shuffle:
            dataset = dataset.shuffle(
                self.shuffle_buffer_size,
                seed=self.seed,
                reshuffle_each_iteration=True
            )
        
        if self.cache:
            dataset = dataset.cache()
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_buffer_size)
        
        return dataset
    
    def get_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        dataset = tf.data.Dataset.from_tensor_slices(
            [str(p) for p in self.file_paths]
        )
        
        dataset = dataset.map(
            self._parse_single_file,
            num_parallel_calls=self.num_parallel_calls
        )
        
        dataset = self._preprocess_dataset(dataset)
        
        n_files = len(self.file_paths)
        n_train = int(n_files * (1 - self.validation_split))
        
        train_dataset = dataset.take(n_train)
        val_dataset = dataset.skip(n_train)
        
        return train_dataset, val_dataset


def create_training_pipeline(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    validation_split: float = 0.2,
    **kwargs
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    pipeline = CosmologicalDataPipeline(
        data_dir=data_dir,
        batch_size=batch_size,
        validation_split=validation_split,
        **kwargs
    )
    
    return pipeline.get_dataset()
```

### 3.2 Data Augmentation

```python
"""Data augmentation strategies for cosmological data."""
import tensorflow as tf
from typing import Optional, Tuple


class CosmologicalAugmenter:
    """Data augmentation for cosmological H(z) data."""
    
    def __init__(
        self,
        noise_std: float = 0.0,
        scale_perturbation: float = 0.0,
        flip_redshift: bool = False,
        random_shift: float = 0.0,
        seed: Optional[int] = None
    ):
        self.noise_std = noise_std
        self.scale_perturbation = scale_perturbation
        self.flip_redshift = flip_redshift
        self.random_shift = random_shift
        self.seed = seed
    
    def augment(self, features: tf.Tensor, target: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.seed is not None:
            tf.random.set_seed(self.seed)
        
        if self.noise_std > 0:
            hz_noise = tf.random.normal(
                shape=[tf.shape(features)[0], 1],
                stddev=self.noise_std,
                dtype=features.dtype
            )
            features = tf.concat([features[:, :1], features[:, 1:] + hz_noise], axis=1)
        
        if self.scale_perturbation > 0:
            scale = tf.random.uniform(
                shape=[],
                minval=1 - self.scale_perturbation,
                maxval=1 + self.scale_perturbation,
                dtype=features.dtype
            )
            features = tf.concat([features[:, :1], features[:, 1:] * scale], axis=1)
            target = target * scale
        
        return features, target
```

---

## 4. Model Architectures

### 4.1 Core Architectures

**File**: `keras_model/models/architectures.py`

```python
"""Standard neural network architectures for cosmological parameter estimation."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Optional, Tuple, List, Dict, Any


class CosmologicalModelFactory:
    """Factory for creating cosmological parameter estimation models."""
    
    @staticmethod
    def build_cnn(
        input_shape: Tuple[int, int] = (80, 2),
        filters: List[int] = [32, 64],
        kernel_sizes: List[int] = [2, 2],
        dense_units: List[int] = [16],
        activation: str = 'relu',
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        output_units: int = 1,
        output_activation: Optional[str] = None,
        name: str = 'cnn_model'
    ) -> keras.Model:
        model = models.Sequential(name=name)
        
        model.add(layers.Input(shape=input_shape))
        
        for i, (filters_i, kernel_i) in enumerate(zip(filters, kernel_sizes)):
            model.add(layers.Conv1D(
                filters=filters_i,
                kernel_size=kernel_i,
                activation=activation,
                padding='same',
                name=f'conv_{i+1}'
            ))
            
            if use_batch_norm:
                model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
            
            model.add(layers.MaxPooling1D(pool_size=2, name=f'pool_{i+1}'))
            
            if dropout > 0:
                model.add(layers.Dropout(dropout, name=f'dropout_{i+1}'))
        
        model.add(layers.Flatten(name='flatten'))
        
        for i, units in enumerate(dense_units):
            model.add(layers.Dense(units, activation=activation, name=f'dense_{i+1}'))
            if dropout > 0:
                model.add(layers.Dropout(dropout, name=f'dropout_dense_{i+1}'))
        
        model.add(layers.Dense(output_units, activation=output_activation, name='output'))
        
        return model
    
    @staticmethod
    def build_dense(
        input_shape: Tuple[int, int] = (80, 2),
        dense_units: List[int] = [16, 16],
        activation: str = 'relu',
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        output_units: int = 1,
        output_activation: Optional[str] = None,
        name: str = 'dense_model'
    ) -> keras.Model:
        model = models.Sequential(name=name)
        
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Flatten(name='flatten'))
        
        for i, units in enumerate(dense_units):
            model.add(layers.Dense(units, activation=activation, name=f'dense_{i+1}'))
            if use_batch_norm:
                model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
            if dropout > 0:
                model.add(layers.Dropout(dropout, name=f'dropout_{i+1}'))
        
        model.add(layers.Dense(output_units, activation=output_activation, name='output'))
        
        return model
    
    @staticmethod
    def build_gru(
        input_shape: Tuple[int, int] = (80, 2),
        units: List[int] = [32],
        bidirectional: bool = False,
        dense_units: List[int] = [16],
        activation: str = 'relu',
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        output_units: int = 1,
        output_activation: Optional[str] = None,
        name: str = 'rnn_model'
    ) -> keras.Model:
        model = models.Sequential(name=name)
        
        model.add(layers.Input(shape=input_shape))
        
        for i, units_i in enumerate(units):
            return_sequences = i < len(units) - 1
            
            if bidirectional:
                model.add(layers.Bidirectional(
                    layers.GRU(
                        units_i,
                        return_sequences=return_sequences,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout
                    ),
                    name=f'bi_gru_{i+1}'
                ))
            else:
                model.add(layers.GRU(
                    units_i,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    name=f'gru_{i+1}'
                ))
        
        for i, units_i in enumerate(dense_units):
            model.add(layers.Dense(units_i, activation=activation, name=f'dense_{i+1}'))
        
        model.add(layers.Dense(output_units, activation=output_activation, name='output'))
        
        return model


def build_cnn_model(**kwargs) -> keras.Model:
    return CosmologicalModelFactory.build_cnn(**kwargs)

def build_dense_model(**kwargs) -> keras.Model:
    return CosmologicalModelFactory.build_dense(**kwargs)

def build_gru_model(**kwargs) -> keras.Model:
    return CosmologicalModelFactory.build_gru(**kwargs)

def build_lstm_model(**kwargs) -> keras.Model:
    return CosmologicalModelFactory.build_lstm(**kwargs)

def build_bidirectional_rnn_model(**kwargs) -> keras.Model:
    kwargs['bidirectional'] = True
    return CosmologicalModelFactory.build_gru(**kwargs)
```

### 4.2 Reusable Building Blocks

```python
"""Reusable model building blocks and components."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ResidualBlock1D(layers.Layer):
    """1D Residual block for convolutional architectures."""
    
    def __init__(self, filters: int, kernel_size: int = 3, activation: str = 'relu', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.act = layers.Activation(activation)
        self.add = layers.Add()
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        if inputs.shape[-1] != self.filters:
            inputs = layers.Conv1D(self.filters, 1, padding='same')(inputs)
        
        x = self.add([x, inputs])
        x = self.act(x)
        
        return x
    
    def get_config(self):
        return {'filters': self.filters, 'kernel_size': self.kernel_size, 'activation': self.activation}


class AttentionBlock1D(layers.Layer):
    """Self-attention block for 1D sequences."""
    
    def __init__(self, units: int, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=units)
        self.norm = layers.LayerNormalization()
        self.add = layers.Add()
    
    def call(self, inputs):
        attn_output = self.attention(inputs, inputs)
        x = self.add([inputs, attn_output])
        x = self.norm(x)
        return x
```

---

## 5. Training Infrastructure

### 5.1 Training Loop

**File**: `keras_model/training/train.py`

```python
"""Training utilities with reproducibility and best practices."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks as keras_callbacks
from typing import Optional, Dict, List, Tuple, Any, Union
from pathlib import Path
import time
import json

from keras_model.utils.seeds import set_seed


def train_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: Optional[tf.data.Dataset] = None,
    epochs: int = 100,
    batch_size: int = 32,
    optimizer: Optional[keras.optimizers.Optimizer] = None,
    loss: Optional[Union[str, keras.losses.Loss]] = None,
    metrics: Optional[List[Union[str, keras.metrics.Metric]]] = None,
    callbacks: Optional[List[keras.callbacks.Callback]] = None,
    seed: Optional[int] = 42,
    verbose: int = 1,
    save_dir: Optional[Union[str, Path]] = None,
    experiment_name: Optional[str] = None,
) -> Tuple[keras.Model, Dict[str, Any]]:
    """Train a Keras model with best practices."""
    if seed is not None:
        set_seed(seed)
    
    if optimizer is None:
        optimizer = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    
    if loss is None:
        loss = keras.losses.MeanSquaredError()
    
    if metrics is None:
        metrics = [
            keras.metrics.MeanAbsoluteError(name='mae'),
            keras.metrics.RootMeanSquaredError(name='rmse'),
        ]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if callbacks is None:
            callbacks = []
        
        callbacks.append(keras_callbacks.ModelCheckpoint(
            filepath=str(save_dir / 'best_model.keras'),
            monitor='val_loss' if val_dataset else 'loss',
            save_best_only=True,
            restore_best_weights=True,
            verbose=1
        ))
        
        callbacks.append(keras_callbacks.CSVLogger(
            str(save_dir / 'training_log.csv'),
            separator=',',
            append=False
        ))
    
    if experiment_name is not None and save_dir is not None:
        exp_dir = save_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            'experiment_name': experiment_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'seed': seed,
        }
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    
    training_time = time.time() - start_time
    history.history['training_time'] = training_time
    
    return model, history.history


def train_with_cross_validation(
    model_fn,
    X: tf.data.Dataset,
    n_splits: int = 5,
    epochs: int = 100,
    seed: int = 42,
    **kwargs
) -> List[Dict[str, Any]]:
    """Train model with k-fold cross-validation."""
    from sklearn.model_selection import KFold
    import numpy as np
    
    set_seed(seed)
    
    all_features = []
    all_targets = []
    for feat, targ in X:
        all_features.append(feat.numpy())
        all_targets.append(targ.numpy())
    
    X_array = np.array(all_features)
    y_array = np.array(all_targets)
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    fold_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_array, y_array)):
        train_data = tf.data.Dataset.from_tensor_slices(
            (X_array[train_idx], y_array[train_idx])
        ).batch(kwargs.get('batch_size', 32))
        
        val_data = tf.data.Dataset.from_tensor_slices(
            (X_array[val_idx], y_array[val_idx])
        ).batch(kwargs.get('batch_size', 32))
        
        model = model_fn()
        
        _, fold_history = train_model(
            model, train_data, val_data,
            epochs=epochs,
            seed=seed + fold,
            experiment_name=f'fold_{fold}',
            **kwargs
        )
        
        fold_histories.append(fold_history)
    
    return fold_histories
```

### 5.2 Custom Callbacks

```python
"""Custom callbacks for training."""
import tensorflow as tf
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
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
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
                import numpy as np
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
```

### 5.3 Custom Losses

```python
"""Custom loss functions for cosmological parameter estimation."""
import tensorflow as tf
from tensorflow import keras
import numpy as np


class HubbleLoss(keras.losses.Loss):
    """Custom loss for Hubble constant estimation."""
    
    def __init__(
        self,
        relative_weight: float = 0.0,
        physics_penalty: float = 0.0,
        reduction: str = keras.losses.Reduction.AUTO,
        name: str = 'hubble_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.relative_weight = relative_weight
        self.physics_penalty = physics_penalty
    
    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        if self.relative_weight > 0:
            rel_error = tf.reduce_mean(tf.square((y_true - y_pred) / (y_true + 1e-8)))
            mse = mse + self.relative_weight * rel_error
        
        if self.physics_penalty > 0:
            penalty = self.physics_penalty * tf.reduce_mean(
                tf.maximum(0.0, -y_pred)
            )
            mse = mse + penalty
        
        return mse
    
    def get_config(self):
        return {
            'relative_weight': self.relative_weight,
            'physics_penalty': self.physics_penalty,
            'name': self.name,
            'reduction': self.reduction,
        }


class QuantileLoss(keras.losses.Loss):
    """Quantile loss for prediction intervals."""
    
    def __init__(
        self,
        quantile: float = 0.5,
        reduction: str = keras.losses.Reduction.AUTO,
        name: str = 'quantile_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.quantile = quantile
    
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        loss = tf.maximum(
            self.quantile * error,
            (self.quantile - 1) * error
        )
        return tf.reduce_mean(loss)
```

---

## 6. Experiment Tracking & Reproducibility

### 6.1 Seed Management

**File**: `keras_model/utils/seeds.py`

```python
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
    """Set random seeds for reproducibility across all libraries."""
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
```

### 6.2 Experiment Tracker

```python
"""Experiment tracking for reproducible research."""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib


class ExperimentTracker:
    """Lightweight experiment tracker for model training."""
    
    def __init__(
        self,
        experiment_dir: Optional[Path] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if experiment_dir is None:
            experiment_dir = Path('experiments')
        
        self.experiment_dir = Path(experiment_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_id = hashlib.md5(
            f"{timestamp}_{time.time()}".encode()
        ).hexdigest()[:8]
        
        if experiment_name is None:
            experiment_name = f"exp_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_path = self.experiment_dir / f"{experiment_name}_{self.experiment_id}"
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        
        self.tags = tags or {}
        self.metadata = metadata or {}
        self.metadata['start_time'] = timestamp
        self.metadata['experiment_id'] = self.experiment_id
        
        self.metrics = {}
        
        self._save_config()
    
    def log_params(self, params: Dict[str, Any]) -> None:
        self.params = params
        self._save_config()
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        if name not in self.metrics:
            self.metrics[name] = {'values': [], 'steps': []}
        
        self.metrics[name]['values'].append(float(value))
        self.metrics[name]['steps'].append(step if step is not None else len(self.metrics[name]['values']))
        
        self._save_metrics()
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_model(self, model_path: Path, metric_value: Optional[float] = None) -> None:
        import shutil
        
        model_dest = self.experiment_path / 'models'
        model_dest.mkdir(exist_ok=True)
        
        model_name = f"model_{len(list(model_dest.glob('model_*')))}"
        if metric_value is not None:
            model_name += f"_val{metric_value:.4f}"
        model_name += '.keras'
        
        shutil.copy2(model_path, model_dest / model_name)
    
    def get_best_epoch(self, metric: str = 'val_loss') -> int:
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} not found")
        
        values = self.metrics[metric]['values']
        return int(np.argmin(values)) + 1
    
    def _save_config(self) -> None:
        config = {
            'experiment_name': self.experiment_name,
            'experiment_id': self.experiment_id,
            'experiment_path': str(self.experiment_path),
            'tags': self.tags,
            'metadata': self.metadata,
        }
        
        if hasattr(self, 'params'):
            config['params'] = self.params
        
        with open(self.experiment_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def _save_metrics(self) -> None:
        with open(self.experiment_path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
```

### 6.3 Model Registry

```python
"""Model registry for tracking trained models."""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import shutil


class ModelRegistry:
    """Registry for trained models with versioning and metadata."""
    
    def __init__(self, registry_dir: Optional[Path] = None):
        if registry_dir is None:
            registry_dir = Path('models')
        
        self.registry_dir = Path(registry_dir)
        self.registry_file = self.registry_dir / 'registry.json'
        
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {'models': []}
    
    def register_model(
        self,
        model_path: Path,
        model_name: str,
        architecture: str,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
        parent_model: Optional[str] = None,
    ) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_id = f"{model_name}_{timestamp}"
        
        model_dir = self.registry_dir / version_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = model_dir / 'model.keras'
        shutil.copy2(model_path, dest_path)
        
        metadata = {
            'version_id': version_id,
            'model_name': model_name,
            'architecture': architecture,
            'metrics': metrics,
            'config': config,
            'tags': tags or {},
            'parent_model': parent_model,
            'registered_at': timestamp,
            'model_path': str(dest_path),
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.registry['models'].append(metadata)
        self._save_registry()
        
        return version_id
    
    def get_model(self, version_id: str) -> Path:
        for model in self.registry['models']:
            if model['version_id'] == version_id:
                return Path(model['model_path'])
        
        raise KeyError(f"Model {version_id} not found in registry")
    
    def list_models(
        self,
        architecture: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        sort_by: Optional[str] = None,
        ascending: bool = True,
    ) -> List[Dict[str, Any]]:
        models = self.registry['models']
        
        if architecture is not None:
            models = [m for m in models if m['architecture'] == architecture]
        
        if tags is not None:
            def matches_tags(m):
                for key, value in tags.items():
                    if m.get('tags', {}).get(key) != value:
                        return False
                return True
            models = [m for m in models if matches_tags(m)]
        
        if sort_by is not None:
            def get_metric(m):
                return m.get('metrics', {}).get(sort_by, float('inf'))
            models = sorted(models, key=get_metric, reverse=not ascending)
        
        return models
    
    def _save_registry(self) -> None:
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        registry_copy = {
            'models': [
                {k: v for k, v in m.items() if k != 'model_path'}
                for m in self.registry['models']
            ]
        }
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry_copy, f, indent=2)
```

---

## 7. Evaluation & Metrics

### 7.1 Custom Metrics

```python
"""Custom metrics for cosmological parameter estimation."""
import tensorflow as tf
from tensorflow import keras
import numpy as np


class MeanAbsolutePercentageError(keras.metrics.Metric):
    """Mean Absolute Percentage Error (MAPE)."""
    
    def __init__(self, name: str = 'mape', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs((y_true - y_pred) / (y_true + 1e-8))
        self.total.assign_add(tf.reduce_sum(error))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def result(self):
        return self.total / (self.count + 1e-8)
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class CoverageProbability(keras.metrics.Metric):
    """Coverage probability for prediction intervals."""
    
    def __init__(
        self,
        lower_percentile: float = 16,
        upper_percentile: float = 84,
        name: str = 'coverage',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
    
    def update_state(self, y_true, y_pred, y_std, sample_weight=None):
        lower = y_pred - self.lower_percentile / 100 * y_std
        upper = y_pred + self.upper_percentile / 100 * y_std
        
        within_bounds = tf.logical_and(
            y_true >= lower,
            y_true <= upper
        )
        
        self.correct = tf.reduce_sum(tf.cast(within_bounds, tf.float32))
        self.total = tf.cast(tf.shape(y_true)[0], tf.float32)
    
    def result(self):
        return self.correct / (self.total + 1e-8)
    
    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
```

### 7.2 Uncertainty Quantification

```python
"""Uncertainty quantification for deep learning models."""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Tuple, Optional, List
from tqdm import tqdm


class MonteCarloDropout:
    """Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(
        self,
        model: keras.Model,
        n_iterations: int = 30,
        dropout_rate: float = 0.1
    ):
        self.model = model
        self.n_iterations = n_iterations
        self.dropout_rate = dropout_rate
    
    def predict(
        self,
        X: tf.data.Dataset,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.trainable = False
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Dropout):
                layer.rate = self.dropout_rate
        
        predictions = []
        
        if verbose:
            iterator = tqdm(range(self.n_iterations), desc='MC Dropout')
        else:
            iterator = range(self.n_iterations)
        
        for _ in iterator:
            preds = self.model.predict(X, verbose=0)
            predictions.append(preds)
        
        predictions = np.array(predictions)
        
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Dropout):
                layer.rate = 0.0
        
        self.model.trainable = True
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred


class EnsemblePredictor:
    """Ensemble predictor for uncertainty estimation."""
    
    def __init__(self, models: List[keras.Model]):
        self.models = models
    
    def predict(
        self,
        X: np.ndarray,
        batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        
        for model in self.models:
            preds = model.predict(X, batch_size=batch_size, verbose=0)
            predictions.append(preds)
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred


def calculate_prediction_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    confidence_levels: List[float] = [0.68, 0.95, 0.99]
) -> dict:
    """Calculate prediction intervals at various confidence levels."""
    from scipy import stats
    
    results = {}
    
    for conf in confidence_levels:
        z = stats.norm.ppf((1 + conf) / 2)
        
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std
        
        within_bounds = np.logical_and(
            y_true >= lower,
            y_true <= upper
        )
        coverage = np.mean(within_bounds)
        
        width = np.mean(upper - lower)
        
        results[f'{int(conf*100)}_percent'] = {
            'coverage': coverage,
            'expected_coverage': conf,
            'mean_width': width,
            'lower_bound': lower,
            'upper_bound': upper,
        }
    
    return results


def sharpness(y_std: np.ndarray) -> float:
    """Calculate sharpness of predictions (average std)."""
    return float(np.mean(y_std))
```

---

## 8. Hyperparameter Management

### 8.1 Hyperparameter Search

```python
"""Hyperparameter search utilities."""
import itertools
import random
from typing import Dict, List, Any, Callable, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm


class HyperparameterSearch:
    """Simple hyperparameter search manager."""
    
    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        search_type: str = 'grid',
        n_iterations: Optional[int] = None,
        seed: int = 42,
    ):
        self.param_grid = param_grid
        self.search_type = search_type
        self.n_iterations = n_iterations
        self.seed = seed
        
        if search_type == 'grid':
            self.param_combinations = self._generate_grid()
        elif search_type == 'random':
            self.param_combinations = self._generate_random()
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    def _generate_grid(self) -> List[Dict[str, Any]]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def _generate_random(self) -> List[Dict[str, Any]]:
        random.seed(self.seed)
        
        combinations = []
        for _ in range(self.n_iterations):
            combination = {}
            for key, values in self.param_grid.items():
                combination[key] = random.choice(values)
            combinations.append(combination)
        
        return combinations
    
    def search(
        self,
        train_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        progress: bool = True,
        save_dir: Optional[Path] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if progress:
            iterator = tqdm(self.param_combinations, desc='Hyperparameter Search')
        else:
            iterator = self.param_combinations
        
        results = []
        best_score = float('-inf')
        best_params = None
        
        for params in iterator:
            metrics = train_fn(params)
            
            result = {
                'params': params,
                'metrics': metrics,
            }
            results.append(result)
            
            score = list(metrics.values())[0]
            if score > best_score:
                best_score = score
                best_params = params
            
            if progress:
                iterator.set_postfix({'best_score': f'{best_score:.4f}'})
            
            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                with open(save_dir / 'search_results.json', 'w') as f:
                    json.dump(results, f, indent=2, default=str)
        
        return best_params, results
```

---

## 9. Implementation Order

### Phase 1: Foundation (Week 1-2)
**Priority**: Critical

1. Create package structure (`keras_model/`)
2. Implement seed management (`utils/seeds.py`)
3. Create basic data pipeline (`data/pipelines.py`)
4. Add model factory (`models/architectures.py`)

### Phase 2: Training (Week 2-3)
**Priority**: High

1. Implement training loop (`training/train.py`)
2. Create standard callbacks (`training/callbacks.py`)
3. Add custom losses (`training/losses.py`)
4. Implement experiment tracker (`experiments/tracker.py`)

### Phase 3: Evaluation (Week 3-4)
**Priority**: High

1. Add custom metrics (`evaluation/metrics.py`)
2. Implement uncertainty quantification (`evaluation/uncertainty.py`)
3. Create model registry (`experiments/registry.py`)
4. Add hyperparameter search (`experiments/search.py`)

### Phase 4: Advanced Features (Week 4-6)
**Priority**: Medium

1. Add data augmentation (`data/augmentation.py`)
2. Create reusable blocks (`models/blocks.py`)
3. Implement ensemble methods
4. Add model comparison tools

### Phase 5: Integration & Testing (Week 6-8)
**Priority**: Medium

1. Test with existing notebooks
2. Create example scripts
3. Generate comparative benchmarks
4. Update documentation

---

## 10. Code Examples

### Example 1: Basic Training with tf.data

```python
import tensorflow as tf
from keras_model.data import CosmologicalDataPipeline
from keras_model.models import build_cnn_model
from keras_model.training import train_model
from keras_model.utils.seeds import set_seed

# Set seed for reproducibility
set_seed(42)

# Create data pipeline
pipeline = CosmologicalDataPipeline(
    data_dir='input/',
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
)

train_ds, val_ds = pipeline.get_dataset()

# Build model
model = build_cnn_model(
    input_shape=(80, 2),
    filters=[32, 64],
    kernel_sizes=[3, 3],
    dense_units=[16],
    dropout=0.2,
)

# Train
model, history = train_model(
    model,
    train_ds,
    val_ds,
    epochs=100,
    save_dir='models/cnn_experiment',
    experiment_name='cnn_h0_v1',
)

print(f"Best validation loss: {min(history['val_loss']):.4f}")
```

### Example 2: Hyperparameter Search

```python
from keras_model.experiments import HyperparameterSearch, ModelRegistry
from keras_model.models import build_cnn_model

# Define search space
param_grid = {
    'filters': [[32, 64], [64, 128]],
    'kernel_sizes': [[2, 2], [3, 3]],
    'dense_units': [[16], [32, 16]],
    'dropout': [0.0, 0.1, 0.2],
    'learning_rate': [1e-3, 1e-4],
}

# Create search
search = HyperparameterSearch(param_grid, search_type='grid')

# Define objective function
def train_with_params(params):
    model = build_cnn_model(input_shape=(80, 2), **params)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    history = model.fit(train_ds, val_ds, epochs=50, verbose=0)
    
    return {'val_loss': min(history.history['val_loss'])}

# Run search
best_params, results = search.search(train_with_params)

print(f"Best params: {best_params}")
```

### Example 3: Uncertainty Quantification

```python
from keras_model.evaluation import MonteCarloDropout, calculate_prediction_intervals

# Wrap model with MC dropout
mc_dropout = MonteCarloDropout(model, n_iterations=50, dropout_rate=0.1)

# Get predictions with uncertainty
y_pred_mean, y_pred_std = mc_dropout.predict(test_ds)

# Calculate prediction intervals
intervals = calculate_prediction_intervals(
    y_test.numpy(),
    y_pred_mean,
    y_pred_std,
    confidence_levels=[0.68, 0.95]
)

print(f"68% coverage: {intervals['68_percent']['coverage']:.2%}")
print(f"95% coverage: {intervals['95_percent']['coverage']:.2%}")
```

### Example 4: Model Registry

```python
from keras_model.experiments import ModelRegistry

registry = ModelRegistry('models')

# Register trained model
version_id = registry.register_model(
    model_path='models/cnn_experiment/best_model.keras',
    model_name='cnn_h0',
    architecture='cnn',
    metrics={'val_loss': 0.001, 'val_mae': 0.5},
    config={'filters': [32, 64], 'dropout': 0.2},
    tags={'target': 'h0', 'dataset': 'lcdm'},
)

# List all CNN models, sorted by validation loss
cnn_models = registry.list_models(
    architecture='cnn',
    sort_by='val_loss',
    ascending=True
)

# Get best model path
best_model_path = registry.get_model(cnn_models[0]['version_id'])
```

---

## 11. Backward Compatibility

### Strategy

1. **New Package Name**: Use `keras_model` (not overwriting existing code)
2. **Legacy Functions**: Keep numpy-based loading in `utils/data_loading.py`
3. **Gradual Adoption**: Existing notebooks work without modification
4. **Opt-In**: New features require explicit import

### Migration Path

```python
# Old way (still works)
from utils.data_loading import load_dataset
X, y = load_dataset('input/')

# New way (tf.data)
from keras_model.data import CosmologicalDataPipeline
pipeline = CosmologicalDataPipeline(data_dir='input/')
train_ds, val_ds = pipeline.get_dataset()
```

---

## Appendix A: Keras Best Practices Checklist

- [ ] Set random seeds before any Model initialization
- [ ] Use `tf.data` pipelines for GPU efficiency
- [ ] Compile with `adam` or `adamw` optimizer
- [ ] Use `MeanSquaredError` loss for regression
- [ ] Track both `loss` and validation metrics
- [ ] Use early stopping with `restore_best_weights=True`
- [ ] Use learning rate reduction on plateau
- [ ] Save models in `.keras` format
- [ ] Log all hyperparameters and metrics
- [ ] Use version control for experiment configs

---

## Appendix B: TensorFlow Version Compatibility

| Feature | TF 2.10 | TF 2.12+ | TF 2.15+ |
|---------|----------|----------|----------|
| `set_random_seed` | ✓ | ✓ (enhanced) | ✓ |
| `tf.data.AUTOTUNE` | ✓ | ✓ | ✓ |
| Mixed precision | ✓ | ✓ | ✓ |
| `.keras` format | ✓ | ✓ | ✓ |
| Deterministic ops | Limited | Better | Best |

**Recommendation**: Use TensorFlow 2.12+ for best reproducibility support.

---

*End of KerasModel TODO Document*
