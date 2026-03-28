"""Utility functions for model analysis and management."""
import numpy as np
from typing import Dict, Any, Optional, List
import json
from pathlib import Path


def save_model_summary(
    model: Any,
    save_path: Optional[Path] = None,
    include_weights: bool = False
) -> Dict[str, Any]:
    """
    Save model architecture summary.
    
    Args:
        model: Keras/TF model or similar
        save_path: Path to save summary
        include_weights: Whether to include weight statistics
        
    Returns:
        Dictionary with model summary
    """
    summary = {
        'model_type': type(model).__name__,
        'total_parameters': model.count_params(),
        'trainable_parameters': sum(
            np.prod(layer.get_weights()[0].shape) 
            for layer in model.layers 
            if layer.get_weights()
        ),
        'non_trainable_parameters': model.count_params() - sum(
            np.prod(layer.get_weights()[0].shape) 
            for layer in model.layers 
            if layer.get_weights()
        ),
        'layers': [],
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }
    
    # Add layer information
    for i, layer in enumerate(model.layers):
        layer_info = {
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': layer.output_shape,
            'parameters': layer.count_params(),
            'trainable': layer.trainable
        }
        
        if include_weights and layer.get_weights():
            weights = layer.get_weights()
            if weights:
                layer_info['weight_shapes'] = [w.shape for w in weights]
                layer_info['weight_statistics'] = [
                    {
                        'mean': float(np.mean(w)),
                        'std': float(np.std(w)),
                        'min': float(np.min(w)),
                        'max': float(np.max(w))
                    }
                    for w in weights
                ]
        
        summary['layers'].append(layer_info)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    return summary


def count_parameters(model: Any) -> Dict[str, int]:
    """
    Count model parameters by layer.
    
    Args:
        model: Keras/TF model
        
    Returns:
        Dictionary with parameter counts
    """
    param_counts = {
        'total': model.count_params(),
        'trainable': 0,
        'non_trainable': 0,
        'by_layer': []
    }
    
    for layer in model.layers:
        layer_params = layer.count_params()
        trainable_params = sum(
            np.prod(w.shape) for w in layer.trainable_weights
        )
        non_trainable_params = layer_params - trainable_params
        
        param_counts['trainable'] += trainable_params
        param_counts['non_trainable'] += non_trainable_params
        
        param_counts['by_layer'].append({
            'layer_name': layer.name,
            'layer_type': layer.__class__.__name__,
            'total_params': layer_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params
        })
    
    return param_counts


def calculate_memory_usage(
    model: Any,
    batch_size: int = 32,
    include_gradients: bool = True
) -> Dict[str, float]:
    """
    Estimate model memory usage.
    
    Args:
        model: Keras/TF model
        batch_size: Batch size for estimation
        include_gradients: Whether to include gradient memory
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Parameter memory
    param_memory = model.count_params() * 4  # Assuming float32 (4 bytes)
    
    # Activation memory (rough estimate)
    activation_memory = 0
    for layer in model.layers:
        if hasattr(layer, 'output_shape'):
            output_shape = layer.output_shape
            if isinstance(output_shape[0], int):
                # Single output shape
                activation_memory += np.prod(output_shape) * 4
            else:
                # Multiple outputs
                for shape in output_shape:
                    if shape is not None:
                        activation_memory += np.prod(shape) * 4
    
    # Gradient memory (same as parameters)
    gradient_memory = param_memory if include_gradients else 0
    
    # Total memory
    total_memory = (param_memory + activation_memory + gradient_memory) * batch_size
    
    # Convert to MB
    memory_mb = total_memory / (1024 * 1024)
    
    return {
        'parameter_memory_mb': param_memory / (1024 * 1024),
        'activation_memory_mb': activation_memory / (1024 * 1024),
        'gradient_memory_mb': gradient_memory / (1024 * 1024),
        'total_memory_mb': memory_mb,
        'per_sample_mb': memory_mb / batch_size
    }


def compare_models(
    models: List[Any],
    model_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare multiple models.
    
    Args:
        models: List of models to compare
        model_names: Names for each model
        
    Returns:
        Comparison dictionary
    """
    if model_names is None:
        model_names = [f'model_{i+1}' for i in range(len(models))]
    
    comparison = {
        'model_names': model_names,
        'total_parameters': [],
        'trainable_parameters': [],
        'input_shapes': [],
        'output_shapes': [],
        'layer_counts': []
    }
    
    for model in models:
        param_counts = count_parameters(model)
        comparison['total_parameters'].append(param_counts['total'])
        comparison['trainable_parameters'].append(param_counts['trainable'])
        comparison['input_shapes'].append(model.input_shape)
        comparison['output_shapes'].append(model.output_shape)
        comparison['layer_counts'].append(len(model.layers))
    
    # Add efficiency metrics if we have multiple models
    if len(models) > 1:
        min_params = min(comparison['total_parameters'])
        max_params = max(comparison['total_parameters'])
        
        comparison['parameter_ratio'] = max_params / min_params if min_params > 0 else float('inf')
    
    return comparison


def extract_model_features(
    model: Any,
    layer_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Extract intermediate features from model layers.
    
    Args:
        model: Keras/TF model
        layer_names: Names of layers to extract from
        
    Returns:
        Dictionary mapping layer names to feature arrays
    """
    from tensorflow.keras.models import Model
    
    if layer_names is None:
        # Extract from all dense/conv layers
        layer_names = [
            layer.name for layer in model.layers 
            if 'dense' in layer.name.lower() or 'conv' in layer.name.lower()
        ]
    
    # Create feature extractor model
    feature_extractor = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(name).output for name in layer_names]
    )
    
    return {
        'layer_names': layer_names,
        'extractor_model': feature_extractor
    }