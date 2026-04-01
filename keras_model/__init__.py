"""keras_model: Deep learning for cosmological parameter estimation."""

__version__ = "0.1.0"

from keras_model.utils.seeds import set_seed
from keras_model.data.pipelines import CosmologicalDataPipeline
from keras_model.models.architectures import (
    create_cnn_model,
    create_dense_model,
    create_gru_model,
    create_lstm_model,
    model_factory,
)
from keras_model.training.train import train_model
from keras_model.training.callbacks import (
    MetricLogger,
    LearningRateSchedulerWithWarmup,
    create_standard_callbacks,
)
from keras_model.training.losses import (
    HubbleLoss,
    QuantileLoss,
    mu_squared_error,
)
from keras_model.experiments.tracker import ExperimentTracker
from keras_model.experiments.registry import ModelRegistry
from keras_model.evaluation.uncertainty import (
    MonteCarloDropout,
    EnsemblePredictor,
)

__all__ = [
    "set_seed",
    "CosmologicalDataPipeline",
    "create_cnn_model",
    "create_dense_model",
    "create_gru_model",
    "create_lstm_model",
    "model_factory",
    "train_model",
    "MetricLogger",
    "LearningRateSchedulerWithWarmup",
    "create_standard_callbacks",
    "HubbleLoss",
    "QuantileLoss",
    "mu_squared_error",
    "ExperimentTracker",
    "ModelRegistry",
    "MonteCarloDropout",
    "EnsemblePredictor",
]
