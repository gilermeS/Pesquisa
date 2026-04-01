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
    train_dataset,
    val_dataset: Optional[Any] = None,
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
    """
    Train a Keras model with best practices.
    
    Args:
        model: Keras model to train
        train_dataset: Training tf.data.Dataset
        val_dataset: Validation tf.data.Dataset (optional)
        epochs: Maximum number of epochs
        batch_size: Batch size
        optimizer: Keras optimizer (default: AdamW)
        loss: Loss function (default: MSE)
        metrics: List of metrics to track
        callbacks: Additional callbacks
        seed: Random seed for reproducibility
        verbose: Verbosity level
        save_dir: Directory to save model checkpoints
        experiment_name: Name for experiment tracking
        
    Returns:
        Tuple of (trained_model, training_history)
    """
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
    
    if hasattr(history, 'history'):
        history.history['training_time'] = training_time
    else:
        history.history = {'training_time': training_time}
    
    return model, history.history


def train_with_cross_validation(
    model_fn,
    X,
    n_splits: int = 5,
    epochs: int = 100,
    seed: int = 42,
    **kwargs
) -> List[Dict[str, Any]]:
    """Train model with k-fold cross-validation."""
    try:
        from sklearn.model_selection import KFold
    except ImportError:
        raise ImportError("sklearn is required for cross-validation")
    import numpy as np
    
    set_seed(seed)
    
    all_features = []
    all_targets = []
    
    try:
        for feat, targ in X:
            all_features.append(feat.numpy())
            all_targets.append(targ.numpy())
    except AttributeError:
        all_features = X[0] if isinstance(X, tuple) else X
        all_targets = X[1] if isinstance(X, tuple) else kwargs.get('y')
    
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
            model,
            train_data,
            val_data,
            epochs=epochs,
            seed=seed + fold,
            experiment_name=f'fold_{fold}',
            **kwargs
        )
        
        fold_histories.append(fold_history)
    
    return fold_histories