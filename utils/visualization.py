"""Visualization utilities for cosmological data analysis."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

from utils.styles import apply_style, get_color_cycle, FIGURE_SIZES, OKABE_ITO
from utils.labels import get_label, format_axis


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training history curves.
    
    Args:
        history: Training history dictionary from model.fit()
        metrics: List of metrics to plot (None = all available)
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Determine metrics to plot
    if metrics is None:
        metrics = [key for key in history.keys() if not key.startswith('val_')]
    
    # Plot training & validation loss
    ax = axes[0]
    if 'loss' in history:
        ax.plot(history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot other metrics
    ax = axes[1]
    other_metrics = [m for m in metrics if m != 'loss']
    
    if other_metrics:
        for metric in other_metrics[:3]:  # Limit to 3 metrics
            ax.plot(history[metric], label=f'Training {metric}', linewidth=2)
            val_metric = f'val_{metric}'
            if val_metric in history:
                ax.plot(history[val_metric], label=f'Validation {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.set_title('Training Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # If no other metrics, plot a blank plot
        ax.text(0.5, 0.5, 'No additional metrics', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        ax.set_title('Additional Metrics')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot true vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('True vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residual plot
    ax = axes[1]
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, alpha=0.6, s=20)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual')
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_cosmological_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    n_samples: int = 5,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot cosmological H(z) data.
    
    Args:
        X: Feature array with shape (n_samples, n_points, 2) for (z, H(z))
        y: Target values (optional)
        n_samples: Number of samples to plot
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot H(z) curves
    ax = axes[0]
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    
    for i, idx in enumerate(indices):
        redshifts = X[idx, :, 0]
        hubble = X[idx, :, 1]
        label = f'Sample {i+1}'
        if y is not None:
            label += f' (H0={y[idx]:.1f})'
        ax.plot(redshifts, hubble, linewidth=2, label=label)
    
    ax.set_xlabel('Redshift (z)')
    ax.set_ylabel('H(z) [km/s/Mpc]')
    ax.set_title('Hubble Parameter vs Redshift')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot distribution of H0 values (if provided)
    ax = axes[1]
    if y is not None:
        ax.hist(y, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(y), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(y):.1f}')
        ax.set_xlabel('Hubble Constant H0 [km/s/Mpc]')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of H0 Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No target values provided', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        ax.set_title('H0 Distribution')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot feature importance as a bar chart.
    
    Args:
        importances: Importance scores for each feature
        feature_names: Names of features
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    # Sort by importance
    indices = np.argsort(importances)
    importances_sorted = importances[indices]
    names_sorted = [feature_names[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(importances_sorted))
    ax.barh(y_pos, importances_sorted)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig