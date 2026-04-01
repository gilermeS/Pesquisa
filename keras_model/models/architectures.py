"""Standard neural network architectures for cosmological parameter estimation."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Optional, Tuple, List


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
        """
        Build 1D CNN model for H(z) data.
        
        Args:
            input_shape: Shape of input data (n_points, n_channels)
            filters: List of filter counts for each Conv1D layer
            kernel_sizes: List of kernel sizes for each Conv1D layer
            dense_units: List of units for each Dense layer
            activation: Activation function
            dropout: Dropout rate
            use_batch_norm: Whether to use BatchNormalization
            output_units: Number of output units
            output_activation: Activation for output layer
            name: Model name
            
        Returns:
            Keras model
        """
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
        """Build fully connected Dense model."""
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
        name: str = 'gru_model'
    ) -> keras.Model:
        """Build RNN model with GRU cells."""
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
    
    @staticmethod
    def build_lstm(
        input_shape: Tuple[int, int] = (80, 2),
        units: List[int] = [32],
        bidirectional: bool = False,
        dense_units: List[int] = [16],
        activation: str = 'relu',
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        output_units: int = 1,
        output_activation: Optional[str] = None,
        name: str = 'lstm_model'
    ) -> keras.Model:
        """Build RNN model with LSTM cells."""
        model = models.Sequential(name=name)
        
        model.add(layers.Input(shape=input_shape))
        
        for i, units_i in enumerate(units):
            return_sequences = i < len(units) - 1
            
            if bidirectional:
                model.add(layers.Bidirectional(
                    layers.LSTM(
                        units_i,
                        return_sequences=return_sequences,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout
                    ),
                    name=f'bi_lstm_{i+1}'
                ))
            else:
                model.add(layers.LSTM(
                    units_i,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    name=f'lstm_{i+1}'
                ))
        
        for i, units_i in enumerate(dense_units):
            model.add(layers.Dense(units_i, activation=activation, name=f'dense_{i+1}'))
        
        model.add(layers.Dense(output_units, activation=output_activation, name='output'))
        
        return model


def build_cnn_model(**kwargs) -> keras.Model:
    """Build CNN model. See CosmologicalModelFactory.build_cnn for args."""
    return CosmologicalModelFactory.build_cnn(**kwargs)

def build_dense_model(**kwargs) -> keras.Model:
    """Build Dense model. See CosmologicalModelFactory.build_dense for args."""
    return CosmologicalModelFactory.build_dense(**kwargs)

def build_gru_model(**kwargs) -> keras.Model:
    """Build GRU model. See CosmologicalModelFactory.build_gru for args."""
    return CosmologicalModelFactory.build_gru(**kwargs)

def build_lstm_model(**kwargs) -> keras.Model:
    """Build LSTM model. See CosmologicalModelFactory.build_lstm for args."""
    return CosmologicalModelFactory.build_lstm(**kwargs)

def build_bidirectional_rnn_model(**kwargs) -> keras.Model:
    """Build bidirectional RNN model. See CosmologicalModelFactory.build_gru for args."""
    kwargs['bidirectional'] = True
    return CosmologicalModelFactory.build_gru(**kwargs)