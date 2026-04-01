"""Custom loss functions for cosmological parameter estimation."""
from typing import Literal
import tensorflow as tf
from tensorflow import keras


class HubbleLoss(keras.losses.Loss):
    """Custom loss for Hubble constant estimation."""
    
    def __init__(
        self,
        relative_weight: float = 0.0,
        physics_penalty: float = 0.0,
        reduction: Literal['auto', 'none', 'sum', 'sum_over_batch_size'] = 'auto',
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
        reduction: Literal['auto', 'none', 'sum', 'sum_over_batch_size'] = 'auto',
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
    
    def get_config(self):
        return {
            'quantile': self.quantile,
            'name': self.name,
            'reduction': self.reduction,
        }


class SymmetricLoss(keras.losses.Loss):
    """Symmetric loss that treats over- and under-prediction equally."""
    
    def __init__(
        self,
        alpha: float = 1.0,
        reduction: Literal['auto', 'none', 'sum', 'sum_over_batch_size'] = 'auto',
        name: str = 'symmetric_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        symmetric_error = tf.abs(error)
        return tf.reduce_mean(tf.pow(symmetric_error, self.alpha))


def mu_squared_error(y_true, y_pred):
    """Wrapper for mean squared error loss."""
    return keras.losses.MeanSquaredError()(y_true, y_pred)