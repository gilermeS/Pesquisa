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
    ax0 = axes[0]
    if 'loss' in history:
        ax0.plot(history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        ax0.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Loss')
    ax0.set_title('Training & Validation Loss')
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    
    # Plot other metrics
    ax1 = axes[1]
    other_metrics = [m for m in metrics if m != 'loss']
    
    if other_metrics:
        for metric in other_metrics[:3]:  # Limit to 3 metrics
            ax1.plot(history[metric], label=f'Training {metric}', linewidth=2)
            val_metric = f'val_{metric}'
            if val_metric in history:
                ax1.plot(history[val_metric], label=f'Validation {metric}', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Metric Value')
        ax1.set_title('Training Metrics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        # If no other metrics, plot a blank plot
        ax1.text(0.5, 0.5, 'No additional metrics', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax1.transAxes)
        ax1.set_title('Additional Metrics')
    
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


def plot_training_history_enhanced(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    smoothing_window: int = 5,
    show_confidence_bands: bool = True,
    show_best_epoch: bool = True,
    show_lr: bool = False,
    language: str = 'both',
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot enhanced training history with smoothing and annotations.
    
    Args:
        history: Training history dict with 'loss', 'val_loss', etc.
        metrics: List of metrics to plot (default: ['loss'])
        smoothing_window: Window size for moving average smoothing
        show_confidence_bands: Show std bands around curves
        show_best_epoch: Mark best epoch with vertical line
        show_lr: Show learning rate on secondary y-axis
        language: 'pt', 'en', or 'both'
        figsize: Figure size (width, height)
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    apply_style('academic')
    colors = get_color_cycle(4)
    
    if metrics is None:
        metrics = ['loss']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    x = np.arange(len(history.get('loss', [0])))
    
    if 'loss' in history and len(history['loss']) > 0:
        loss = np.array(history['loss'])
        val_loss = np.array(history.get('val_loss', []))
        
        smoothed_loss = _smooth_curve(loss, smoothing_window)
        ax.plot(x, loss, alpha=0.3, color=colors[0], linewidth=1)
        ax.plot(x, smoothed_loss, color=colors[0], linewidth=2, label=get_label('train_loss', language))
        
        if show_confidence_bands and len(loss) >= smoothing_window:
            std_loss = _compute_std_bands(loss, smoothing_window)
            ax.fill_between(x, smoothed_loss - std_loss, smoothed_loss + std_loss, alpha=0.2, color=colors[0])
        
        if len(val_loss) > 0:
            smoothed_val = _smooth_curve(val_loss, smoothing_window)
            ax.plot(x, val_loss, alpha=0.3, color=colors[1], linewidth=1)
            ax.plot(x, smoothed_val, color=colors[1], linewidth=2, label=get_label('val_loss', language))
            
            if show_confidence_bands and len(val_loss) >= smoothing_window:
                std_val = _compute_std_bands(val_loss, smoothing_window)
                ax.fill_between(x, smoothed_val - std_val, smoothed_val + std_val, alpha=0.2, color=colors[1])
        
        if show_best_epoch and len(val_loss) > 0:
            best_epoch = int(np.argmin(val_loss))
            ax.axvline(best_epoch, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.annotate(f'Best: {best_epoch}', xy=(best_epoch, val_loss[best_epoch]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8, color='gray')
    
    ax.set_xlabel(get_label('epoch', language))
    ax.set_ylabel(get_label('loss', language))
    ax.set_title(get_label('loss', language))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    other_metrics = [m for m in metrics if m != 'loss']
    
    if other_metrics:
        metric = other_metrics[0]
        if metric in history and len(history[metric]) > 0:
            m = np.array(history[metric])
            smoothed_m = _smooth_curve(m, smoothing_window)
            ax.plot(x, m, alpha=0.3, color=colors[2], linewidth=1)
            ax.plot(x, smoothed_m, color=colors[2], linewidth=2, label=metric)
            
            val_m = f'val_{metric}'
            if val_m in history and len(history[val_m]) > 0:
                val_metric_data = np.array(history[val_m])
                smoothed_val_m = _smooth_curve(val_metric_data, smoothing_window)
                ax.plot(x, val_metric_data, alpha=0.3, color=colors[3], linewidth=1)
                ax.plot(x, smoothed_val_m, color=colors[3], linewidth=2, label=val_m)
    
    if show_lr and 'learning_rate' in history:
        ax_lr = ax.twinx()
        lr = np.array(history['learning_rate'])
        ax_lr.plot(x, lr, color='gray', linewidth=1, alpha=0.7)
        ax_lr.set_ylabel(get_label('learning_rate', language), color='gray')
        ax_lr.tick_params(axis='y', labelcolor='gray')
    
    ax.set_xlabel(get_label('epoch', language))
    ax.set_ylabel(get_label('metric', language))
    ax.set_title(get_label('metric', language))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def _smooth_curve(values: np.ndarray, window: int) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def _compute_std_bands(values: np.ndarray, window: int) -> np.ndarray:
    """Compute standard deviation bands for smoothing."""
    if len(values) < window:
        return np.zeros_like(values)
    result = np.zeros_like(values)
    half = window // 2
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        result[i] = np.std(values[start:end])
    return result


def plot_predictions_enhanced(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    percentiles: Tuple[int, int] = (5, 95),
    show_intervals: Tuple[int, int] = (68, 95),
    color_by: Optional[str] = 'density',
    highlight_outliers: bool = True,
    outlier_threshold: float = 2.0,
    language: str = 'both',
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot enhanced prediction analysis with confidence intervals.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_std: Standard deviation of predictions
        percentiles: Percentiles for confidence intervals
        show_intervals: Confidence levels to display (e.g., 68, 95)
        color_by: Color by 'density', 'residual', or None
        highlight_outliers: Whether to highlight outliers
        outlier_threshold: Threshold in std for outlier detection
        language: 'pt', 'en', or 'both'
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    apply_style('academic')
    colors = get_color_cycle(3)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    residuals = y_pred - y_true
    outliers = np.abs(residuals) > outlier_threshold * np.std(residuals)
    
    if color_by == 'density' and y_std is not None:
        scatter = ax.scatter(y_true, y_pred, c=y_std, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label=get_label('uncertainty', language))
    elif color_by == 'residual':
        scatter = ax.scatter(y_true, y_pred, c=np.abs(residuals), cmap='plasma', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label=get_label('residual', language))
    else:
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, color=colors[0])
    
    if highlight_outliers and np.any(outliers):
        ax.scatter(y_true[outliers], y_pred[outliers], facecolors='none', edgecolors='red', s=80, linewidths=2, label='Outliers')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label=get_label('true_value', language))
    
    if y_std is not None:
        for level in show_intervals:
            z = {68: 1, 95: 2}.get(level, 1)
            ax.fill_between(
                [min_val, max_val],
                [min_val - z * np.mean(y_std), max_val - z * np.mean(y_std)],
                [min_val + z * np.mean(y_std), max_val + z * np.mean(y_std)],
                alpha=0.1, color=colors[1], label=f'{level}% CI'
            )
    
    ax.set_xlabel(get_label('true_value', language))
    ax.set_ylabel(get_label('predicted_value', language))
    ax.set_title(get_label('comparison', language))
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.scatter(y_true, residuals, alpha=0.6, s=20, color=colors[0])
    ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
    
    if y_std is not None:
        for level in show_intervals:
            z = {68: 1, 95: 2}.get(level, 1)
            ax.fill_between(
                [y_true.min(), y_true.max()],
                [-z * np.mean(y_std), -z * np.mean(y_std)],
                [z * np.mean(y_std), z * np.mean(y_std)],
                alpha=0.1, color=colors[1]
            )
    
    if highlight_outliers and np.any(outliers):
        ax.scatter(y_true[outliers], residuals[outliers], facecolors='none', edgecolors='red', s=80, linewidths=2)
    
    ax.set_xlabel(get_label('true_value', language))
    ax.set_ylabel(get_label('residual', language))
    ax.set_title(get_label('residual', language))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    language: str = 'both',
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot residual diagnostics for model evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        language: 'pt', 'en', or 'both'
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    apply_style('academic')
    colors = get_color_cycle(4)
    
    residuals = y_pred - y_true
    standardized = (residuals - np.mean(residuals)) / np.std(residuals)
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.6, s=20, color=colors[0])
    ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
    ax.axhline(y=2 * np.std(residuals), color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y=-2 * np.std(residuals), color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel(get_label('predicted_value', language))
    ax.set_ylabel(get_label('residual', language))
    ax.set_title('Residuals vs Predicted')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    _plot_qq(standardized, ax, colors)
    ax.set_title('Q-Q Plot')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.hist(residuals, bins=30, density=True, alpha=0.7, color=colors[0], edgecolor='black')
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    from scipy.stats import norm
    ax.plot(x_range, norm.pdf(x_range, np.mean(residuals), np.std(residuals)), 'r-', linewidth=2, label='Normal')
    ax.set_xlabel(get_label('residual', language))
    ax.set_ylabel(get_label('frequency', language))
    ax.set_title(get_label('distribution', language))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[3]
    sqrt_abs = np.sqrt(np.abs(standardized))
    ax.scatter(y_pred, sqrt_abs, alpha=0.6, s=20, color=colors[0])
    z = np.polyfit(y_pred, sqrt_abs, 1)
    p = np.poly1d(z)
    ax.plot(y_pred, p(y_pred), 'r--', linewidth=2)
    ax.set_xlabel(get_label('predicted_value', language))
    ax.set_ylabel(r'$\sqrt{|standardized residuals|}$')
    ax.set_title('Scale-Location')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def _plot_qq(data: np.ndarray, ax, colors) -> None:
    """Plot Q-Q plot."""
    from scipy.stats import probplot
    probplot(data, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(colors[0])
    ax.get_lines()[0].set_markeredgecolor(colors[0])
    ax.get_lines()[1].set_color('red')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['r2', 'mae', 'rmse', 'mape'],
    comparison_type: str = 'radar',
    language: str = 'both',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot model comparison with multiple visualization types.
    
    Args:
        results: Dict mapping model names to metric dicts
        metrics: List of metrics to compare
        comparison_type: 'radar', 'box', or 'bar'
        language: 'pt', 'en', or 'both'
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    apply_style('academic')
    colors = get_color_cycle(max(4, len(results)))
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.8 / len(metrics)
    
    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in model_names]
        ax.bar(x + i * width - (len(metrics) - 1) * width / 2, values, width, label=metric, color=colors[i])
    
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel(get_label('value', language))
    ax.set_title(get_label('comparison', language))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1]
    if comparison_type == 'radar':
        _plot_radar_chart(results, metrics, ax, colors)
    elif comparison_type == 'box':
        _plot_box_comparison(results, metrics, ax, colors)
    else:
        ax.text(0.5, 0.5, f'Unknown comparison type: {comparison_type}', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def _plot_radar_chart(results: Dict, metrics: List[str], ax, colors) -> None:
    """Plot radar chart for model comparison."""
    model_names = list(results.keys())
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_ylim(0, 1)
    
    for i, model in enumerate(model_names):
        values = [results[model].get(m, 0) for m in metrics]
        if 'r2' in metrics:
            idx = metrics.index('r2')
            values[idx] = results[model].get('r2', 0) / 2 + 0.5
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))


def _plot_box_comparison(results: Dict, metrics: List[str], ax, colors) -> None:
    """Plot box comparison."""
    model_names = list(results.keys())
    metric = metrics[0]
    values = [[results[m].get(metric, 0)] for m in model_names]
    bp = ax.boxplot(values, labels=model_names, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3, axis='y')


def plot_feature_importance_enhanced(
    importances: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    significance: Optional[np.ndarray] = None,
    normalize: bool = True,
    sort_by: str = 'importance',
    color_scheme: str = 'single',
    language: str = 'both',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot enhanced feature importance with error bars and significance markers.
    
    Args:
        importances: Importance scores for each feature
        uncertainties: Uncertainty (std) for each importance
        feature_names: Names of features
        significance: Significance levels (e.g., p-values)
        normalize: Whether to normalize importances to sum to 1
        sort_by: 'importance' or 'name'
        color_scheme: 'single', 'significance', or 'gradient'
        language: 'pt', 'en', or 'both'
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    apply_style('academic')
    colors = get_color_cycle(4)
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    imp = np.array(importances)
    if normalize:
        imp = imp / imp.sum() * 100
    
    indices = np.argsort(imp)
    if sort_by == 'importance':
        indices = indices[::-1]
    imp_sorted = imp[indices]
    names_sorted = [feature_names[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(imp_sorted))
    
    if color_scheme == 'significance' and significance is not None:
        sig_sorted = np.array(significance)[indices]
        color_vals = sig_sorted
        cmap = plt.cm.RdYlGn_r
        sc = ax.barh(y_pos, imp_sorted, color=cmap(color_vals / color_vals.max()))
    elif color_scheme == 'gradient':
        gradient = np.linspace(0, 1, len(imp_sorted))
        ax.barh(y_pos, imp_sorted, color=plt.cm.viridis(gradient))
    else:
        ax.barh(y_pos, imp_sorted, color=colors[0])
    
    if uncertainties is not None:
        unc_sorted = np.array(uncertainties)[indices]
        if normalize:
            unc_sorted = unc_sorted / np.sum(uncertainties) * 100
        ax.errorbar(imp_sorted, y_pos, xerr=unc_sorted, fmt='none', color='black', capsize=2)
    
    if significance is not None:
        for i, (imp_val, sig_val) in enumerate(zip(imp_sorted, np.array(significance)[indices])):
            if sig_val < 0.001:
                marker = '***'
            elif sig_val < 0.01:
                marker = '**'
            elif sig_val < 0.05:
                marker = '*'
            else:
                marker = ''
            if marker:
                ax.annotate(marker, xy=(imp_val, i), xytext=(5, 0), textcoords='offset points', fontsize=10, va='bottom')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted)
    ax.set_xlabel(get_label('relative_importance', language))
    ax.set_title(get_label('feature_importance', language))
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_timestep_importance_enhanced(
    importances: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    redshifts: Optional[np.ndarray] = None,
    show_confidence: bool = True,
    show_theory: bool = False,
    theory_model: Optional[str] = 'LCDM',
    highlight_regions: Optional[List[Tuple[float, float]]] = None,
    language: str = 'both',
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot enhanced timestep/redshift importance with confidence bands.
    
    Args:
        importances: Importance scores per timestep
        uncertainties: Uncertainty per timestep
        redshifts: Redshift values for x-axis
        show_confidence: Show confidence bands
        show_theory: Overlay theoretical model
        theory_model: Cosmological model name
        highlight_regions: List of (start, end) tuples to highlight
        language: 'pt', 'en', or 'both'
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    apply_style('academic')
    colors = get_color_cycle(3)
    
    if redshifts is None:
        x = np.arange(len(importances))
    else:
        x = np.array(redshifts)
    
    imp = np.array(importances)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if show_confidence and uncertainties is not None:
        unc = np.array(uncertainties)
        ax.fill_between(x, imp - unc, imp + unc, alpha=0.3, color=colors[0], label=get_label('confidence_interval', language))
    
    ax.plot(x, imp, '-o', markersize=4, linewidth=1.5, color=colors[0], label=get_label('timestep_importance', language))
    
    if show_theory:
        theory_z = np.linspace(x.min(), x.max(), 100)
        theory_imp = _compute_theory_importance(theory_z, theory_model)
        ax.plot(theory_z, theory_imp, '--', color=colors[1], linewidth=2, label=theory_model)
    
    if highlight_regions is not None:
        for start, end in highlight_regions:
            ax.axvspan(start, end, alpha=0.2, color=colors[2])
    
    ax.set_xlabel(get_label('redshift', language))
    ax.set_ylabel(get_label('importance', language))
    ax.set_title(get_label('timestep_importance', language))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def _compute_theory_importance(z: np.ndarray, model: str) -> np.ndarray:
    """Compute theoretical importance curve for comparison."""
    if model == 'LCDM':
        return 1.0 / (1 + z) ** 2
    elif model == 'wCDM':
        return 1.0 / (1 + z) ** 3
    else:
        return np.ones_like(z)


def plot_cosmological_data_enhanced(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    y_std: Optional[np.ndarray] = None,
    n_samples: int = 20,
    show_samples: bool = True,
    show_mean: bool = True,
    show_confidence_bands: Tuple[int, int] = (1, 2),
    show_theory: bool = True,
    theory_params: Optional[Dict] = None,
    show_residuals: bool = True,
    residual_height: float = 0.15,
    observational_data: Optional[np.ndarray] = None,
    observational_errors: Optional[np.ndarray] = None,
    language: str = 'both',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot enhanced cosmological H(z) data with theory comparison.
    
    Args:
        X: Feature array with shape (n_samples, n_points, 2) for (z, H(z))
        y: Target values (optional)
        y_std: Standard deviation of H0 estimates
        n_samples: Number of samples to plot
        show_samples: Whether to show individual samples
        show_mean: Whether to show mean curve
        show_confidence_bands: Confidence levels (e.g., (1, 2) for 1σ and 2σ)
        show_theory: Overlay theoretical model
        theory_params: Parameters for theory curve
        show_residuals: Show residuals inset
        residual_height: Height of residuals subplot
        observational_data: Observational H(z) data points
        observational_errors: Errors for observational data
        language: 'pt', 'en', or 'both'
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    apply_style('academic')
    colors = get_color_cycle(4)
    
    fig = plt.figure(figsize=figsize)
    
    if show_residuals:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, residual_height], hspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_resid = fig.add_subplot(gs[1], sharex=ax_main)
    else:
        ax_main = fig.add_subplot(111)
    
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    
    if show_samples:
        for idx in indices:
            redshifts = X[idx, :, 0]
            hubble = X[idx, :, 1]
            ax_main.plot(redshifts, hubble, alpha=0.15, color=colors[0], linewidth=0.5)
    
    if y is not None:
        all_z = X[:, :, 0].mean(axis=0)
        all_h = X[:, :, 1]
        
        mean_h = all_h.mean(axis=0)
        std_h = all_h.std(axis=0)
        
        if show_mean:
            ax_main.plot(all_z, mean_h, color=colors[1], linewidth=2, label=get_label('observation', language))
        
        for n_sig in show_confidence_bands:
            ax_main.fill_between(all_z, mean_h - n_sig * std_h, mean_h + n_sig * std_h, alpha=0.15, color=colors[1])
    
    if show_theory:
        z_theory = np.linspace(0, max(all_z.max(), 2), 100)
        H_theory = _compute_theory_hubble(z_theory, theory_params)
        ax_main.plot(z_theory, H_theory, '--', color=colors[2], linewidth=2, label=get_label('theory', language))
    
    if observational_data is not None and len(observational_data) > 0:
        z_obs, h_obs = observational_data[:2]
        ax_main.errorbar(z_obs, h_obs, yerr=observational_errors[2] if observational_errors is not None else None,
                        fmt='o', color='black', capsize=3, markersize=6, label=get_label('observation', language), zorder=10)
    
    ax_main.set_ylabel(get_label('hubble', language))
    ax_main.legend(loc='upper left')
    ax_main.grid(True, alpha=0.3)
    
    if show_residuals:
        ax_resid.set_xlabel(get_label('redshift', language))
        ax_resid.set_ylabel(get_label('residual', language))
        
        if y is not None and show_theory:
            residuals = mean_h - H_theory[:len(mean_h)]
            ax_resid.scatter(all_z, residuals, color=colors[1], s=20)
            ax_resid.axhline(y=0, color='gray', linestyle='--')
            ax_resid.grid(True, alpha=0.3)
        
        plt.setp(ax_main.get_xticklabels(), visible=False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def _compute_theory_hubble(z: np.ndarray, params: Optional[Dict] = None) -> np.ndarray:
    """Compute theoretical H(z) curve."""
    if params is None:
        params = {'H0': 67.45, 'Omega_m': 0.315}
    
    H0 = params.get('H0', 67.45)
    Om = params.get('Omega_m', 0.315)
    
    return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))