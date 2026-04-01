"""Uncertainty quantification for deep learning models."""
import numpy as np
from typing import Tuple, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from tensorflow import keras


class MonteCarloDropout:
    """Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(
        self,
        model,
        n_iterations: int = 30,
        dropout_rate: float = 0.1
    ):
        """
        Initialize MC Dropout wrapper.
        
        Args:
            model: Trained Keras model with dropout layers
            n_iterations: Number of forward passes with dropout
            dropout_rate: Dropout rate to use at inference
        """
        self.model = model
        self.n_iterations = n_iterations
        self.dropout_rate = dropout_rate
    
    def predict(self, X, batch_size: Optional[int] = None, verbose: bool = False):
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Input data (numpy array or tf.data.Dataset)
            batch_size: Batch size for prediction
            verbose: Show progress
            
        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False
            verbose = False
        
        try:
            import tensorflow as tf
            has_tf = True
        except ImportError:
            has_tf = False
        
        if has_tf:
            self.model.trainable = False
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Dropout):
                    layer.rate = self.dropout_rate
        
        predictions = []
        
        if verbose and has_tqdm:
            iterator = tqdm(range(self.n_iterations), desc='MC Dropout')
        else:
            iterator = range(self.n_iterations)
        
        for _ in iterator:
            if has_tf and hasattr(self.model, 'predict'):
                preds = self.model.predict(X, batch_size=batch_size, verbose=0)
            else:
                preds = self.model(X)
                if hasattr(preds, 'numpy'):
                    preds = preds.numpy()
            predictions.append(preds.flatten() if len(preds.shape) > 1 else preds)
        
        predictions = np.array(predictions)
        
        if has_tf:
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Dropout):
                    layer.rate = 0.0
            self.model.trainable = True
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred


class EnsemblePredictor:
    """Ensemble predictor for uncertainty estimation."""
    
    def __init__(self, models: List):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained Keras models
        """
        self.models = models
    
    def predict(
        self,
        X,
        batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions with uncertainty."""
        predictions = []
        
        for model in self.models:
            preds = model.predict(X, batch_size=batch_size, verbose=0)
            predictions.append(preds.flatten() if len(preds.shape) > 1 else preds)
        
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
    """
    Calculate prediction intervals at various confidence levels.
    
    Args:
        y_true: True values
        y_pred: Predicted mean values
        y_std: Predicted standard deviations
        confidence_levels: List of confidence levels (as fractions)
        
    Returns:
        Dictionary with interval information
    """
    try:
        from scipy import stats
        has_scipy = True
    except ImportError:
        has_scipy = False
    
    results = {}
    
    for conf in confidence_levels:
        if has_scipy:
            z = stats.norm.ppf((1 + conf) / 2)
        else:
            import math
            z = math.sqrt(2) * math.erfinv(conf)
        
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
    """
    Calculate sharpness of predictions.
    
    Sharpness is the average standard deviation of predictions.
    Lower is better (more precise).
    """
    return float(np.mean(y_std))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def calculate_mae_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAE as percentage of true value."""
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-8)) * 100