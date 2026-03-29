# GraphTODO - Visualization Improvement Plan

This document outlines a comprehensive plan for improving the visualization of results in the cosmological data analysis project. Created for implementation by AI coding agents and developers.

**Last Updated**: March 2026  
**Status**: Planned (not yet implemented)  
**Priority**: Medium-High  
**Estimated Effort**: 4-6 weeks

---

## Table of Contents

1. [Overview](#overview)
2. [Foundation: Style and Consistency System](#1-foundation-style-and-consistency-system)
3. [Model Performance Visualizations](#2-model-performance-visualizations)
4. [Feature Importance Visualizations](#3-feature-importance-visualizations)
5. [Cosmological Data Visualizations](#4-cosmological-data-visualizations)
6. [Implementation Order](#5-implementation-order)
7. [Code Examples](#6-code-examples)
8. [Backward Compatibility](#7-backward-compatibility)

---

## Overview

### Current State Analysis

**Existing Visualizations**:
- Training loss curves (simple line plots)
- Prediction scatter plots
- Permutation importance (bar charts, colored timestep plots)
- H(z) data curves
- Parameter distribution histograms
- Model architecture diagrams

**Issues Identified**:
1. Inconsistent styling across notebooks
2. Limited uncertainty visualization
3. No confidence intervals on predictions
4. Portuguese-only labels (limited accessibility)
5. Missing statistical annotations
6. No publication-ready formatting
7. Limited model comparison views
8. Missing residual diagnostics

### Goals

1. **Publication Quality**: Generate figures ready for academic papers and theses
2. **Consistency**: Unified style across all visualizations
3. **Accessibility**: Colorblind-friendly palettes, bilingual labels (Portuguese/English)
4. **Insight Depth**: Better uncertainty quantification and statistical analysis
5. **Reusability**: Modular functions for future projects

---

## 1. Foundation: Style and Consistency System

### 1.1 Create Style Module

**File**: `utils/styles.py` (new)

**Purpose**: Centralized style management for all plots

**Contents**:

```python
"""Style configuration for academic visualizations."""
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any

# Color palettes
OKABE_ITO = [
    '#0072B2',  # Blue
    '#E69F00',  # Orange
    '#009E73',  # Green
    '#CC79A7',  # Pink
    '#56B4E9',  # Sky Blue
    '#D55E00',  # Vermillion
    '#F0E442',  # Yellow
    '#000000',  # Black
]

COLORBLIND_SAFE = OKABE_ITO

SEQUENTIAL_PALETTES = {
    'viridis': 'viridis',
    'plasma': 'plasma',
    'inferno': 'inferno',
    'magma': 'magma',
}

# Paper-quality figure sizes (in inches)
FIGURE_SIZES = {
    'single': (3.5, 2.625),      # ~90mm width (single column)
    'double': (7.0, 5.25),        # ~180mm width (double column)
    'full': (7.0, 5.25),         # Full page width
    'square': (5.0, 5.0),         # Square aspect ratio
    'wide': (10.0, 4.0),         # Wide format for timelines
    'tall': (5.0, 7.0),          # Tall format for distributions
}

ACADEMIC_STYLE = {
    # Font settings
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.title_fontsize': 10,
    
    # Line and marker settings
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'errorbar.capsize': 2,
    
    # Grid settings
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    
    # Figure settings
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.format': 'png',
    
    # Spine settings
    'axes.spines.top': False,
    'axes.spines.right': False,
}

PRESENTER_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'lines.linewidth': 2,
    'figure.figsize': (12, 7),
}


def apply_style(style: str = 'academic') -> None:
    """
    Apply a predefined style to matplotlib.
    
    Args:
        style: One of 'academic', 'presenter', 'dark'
    """
    if style == 'academic':
        plt.rcParams.update(ACADEMIC_STYLE)
    elif style == 'presenter':
        plt.rcParams.update(PRESENTER_STYLE)
    elif style == 'dark':
        _apply_dark_style()
    else:
        raise ValueError(f"Unknown style: {style}")


def reset_style() -> None:
    """Reset matplotlib to default settings."""
    plt.rcParams.update(plt.rcParamsDefault)
```

### 1.2 Bilingual Label System

**File**: `utils/labels.py` (new)

**Purpose**: Support Portuguese and English labels for international accessibility

```python
"""Bilingual label system for visualizations."""
from typing import Dict, Optional, Union


class BilingualLabel:
    """Manages bilingual labels for plot elements."""
    
    def __init__(self, pt: str, en: str):
        self.pt = pt
        self.en = en
    
    def get(self, language: str = 'both') -> str:
        """
        Get label in requested language format.
        
        Args:
            language: 'pt', 'en', or 'both'
            
        Returns:
            Label string in requested format
        """
        if language == 'pt':
            return self.pt
        elif language == 'en':
            return self.en
        elif language == 'both':
            return f"{self.pt}\n({self.en})"
        else:
            raise ValueError(f"Unknown language: {language}")


# Label definitions
LABELS: Dict[str, BilingualLabel] = {
    # Cosmological parameters
    'redshift': BilingualLabel('Deslocamento para o vermelho (z)', 'Redshift (z)'),
    'hubble': BilingualLabel('Parâmetro de Hubble H(z)', 'Hubble Parameter H(z)'),
    'hubble_unit': BilingualLabel('H(z) [km s⁻¹ Mpc⁻¹]', 'H(z) [km s⁻¹ Mpc⁻¹]'),
    'h0': BilingualLabel('Constante de Hubble H₀', 'Hubble Constant H₀'),
    'h0_unit': BilingualLabel('H₀ [km s⁻¹ Mpc⁻¹]', 'H₀ [km s⁻¹ Mpc⁻¹]'),
    'omega_m': BilingualLabel('Parâmetro de densidade de matéria Ωₘ', 'Matter density parameter Ωₘ'),
    'omega_de': BilingualLabel('Parâmetro de densidade de energia escura Ω₌', 'Dark energy density parameter Ω₌'),
    'w0': BilingualLabel('Parâmetro EoS da energia escura w₀', 'Dark energy EoS parameter w₀'),
    'wa': BilingualLabel('Parâmetro de evolução da EoS wₐ', 'EoS evolution parameter wₐ'),
    
    # Model metrics
    'loss': BilingualLabel('Perda', 'Loss'),
    'mse': BilingualLabel('Erro quadrático médio (MSE)', 'Mean Squared Error (MSE)'),
    'mae': BilingualLabel('Erro absoluto médio (MAE)', 'Mean Absolute Error (MAE)'),
    'rmse': BilingualLabel('Raiz do erro quadrático médio (RMSE)', 'Root Mean Squared Error (RMSE)'),
    'r2': BilingualLabel('Coeficiente de determinação R²', 'Coefficient of determination R²'),
    'val_loss': BilingualLabel('Perda de validação', 'Validation Loss'),
    'train_loss': BilingualLabel('Perda de treino', 'Training Loss'),
    
    # Training
    'epoch': BilingualLabel('Época', 'Epoch'),
    'learning_rate': BilingualLabel('Taxa de aprendizado', 'Learning Rate'),
    'batch_size': BilingualLabel('Tamanho do batch', 'Batch Size'),
    
    # Visualization elements
    'true_value': BilingualLabel('Valor verdadeiro', 'True Value'),
    'predicted_value': BilingualLabel('Valor previsto', 'Predicted Value'),
    'residual': BilingualLabel('Residual', 'Residual'),
    'residuals': BilingualLabel('Residuais', 'Residuals'),
    'uncertainty': BilingualLabel('Incerteza', 'Uncertainty'),
    'confidence_interval': BilingualLabel('Intervalo de confiança', 'Confidence Interval'),
    
    # Feature importance
    'feature_importance': BilingualLabel('Importância das features', 'Feature Importance'),
    'permutation_importance': BilingualLabel('Importância por permutação', 'Permutation Importance'),
    'timestep_importance': BilingualLabel('Importância por passo temporal', 'Timestep Importance'),
    'relative_importance': BilingualLabel('Importância relativa', 'Relative Importance'),
    'decrease_performance': BilingualLabel('Queda de desempenho', 'Performance Decrease'),
    
    # Model names
    'cnn': BilingualLabel('CNN', 'CNN'),
    'dense': BilingualLabel('Densa', 'Dense'),
    'rnn': BilingualLabel('RNN', 'RNN'),
    'rnn_bi': BilingualLabel('RNN Bidirecional', 'Bidirectional RNN'),
    'svm': BilingualLabel('SVM', 'SVM'),
    'lcdm': BilingualLabel('LCDM', 'LCDM'),
    'wcdm': BilingualLabel('wCDM', 'wCDM'),
    'wacdm': BilingualLabel('w(a)CDM', 'w(a)CDM'),
    
    # General
    'sample': BilingualLabel('Amostra', 'Sample'),
    'frequency': BilingualLabel('Frequência', 'Frequency'),
    'distribution': BilingualLabel('Distribuição', 'Distribution'),
    'comparison': BilingualLabel('Comparação', 'Comparison'),
    'theory': BilingualLabel('Teoria', 'Theory'),
    'observation': BilingualLabel('Observação', 'Observation'),
    'simulation': BilingualLabel('Simulação', 'Simulation'),
}


def get_label(key: str, language: str = 'both') -> str:
    """
    Get bilingual label for a key.
    
    Args:
        key: Label key (e.g., 'redshift', 'h0')
        language: 'pt', 'en', or 'both'
        
    Returns:
        Formatted label string
    """
    if key not in LABELS:
        raise KeyError(f"Unknown label key: {key}")
    return LABELS[key].get(language)


def format_axis(ax, x_key: Optional[str] = None, y_key: Optional[str] = None, 
                title_key: Optional[str] = None, language: str = 'both') -> None:
    """
    Format axes with bilingual labels.
    
    Args:
        ax: Matplotlib axis object
        x_key: Label key for x-axis
        y_key: Label key for y-axis
        title_key: Label key for title
        language: Language setting
    """
    if x_key:
        ax.set_xlabel(get_label(x_key, language), fontsize=ax.xaxis.label.get_size())
    if y_key:
        ax.set_ylabel(get_label(y_key, language), fontsize=ax.yaxis.label.get_size())
    if title_key:
        ax.set_title(get_label(title_key, language), fontsize=ax.title.get_size())
```

---

## 2. Model Performance Visualizations

### 2.1 Enhanced Training History

**Current State**: Simple line plots showing loss vs epoch

**Improvements**:
- Moving average smoothing for noisy curves
- Confidence bands (std across folds)
- Best epoch marker
- Learning rate overlay (secondary axis)
- Side-by-side model comparison

**New Function**: `plot_training_history_enhanced()`

```python
def plot_training_history_enhanced(
    history: Dict[str, list],
    metrics: Optional[List[str]] = None,
    smoothing_window: int = 5,
    show_confidence_bands: bool = True,
    show_best_epoch: bool = True,
    show_lr: bool = False,
    language: str = 'both',
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None
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
        
    Returns:
        Matplotlib figure
    """
```

**Visual Design**:
- Two-column layout: Loss | Additional metrics
- Smoothed lines (solid) overlaid with raw data (faded)
- Shaded confidence bands (α=0.2)
- Vertical dashed line at best epoch
- Legend with metric names

### 2.2 Prediction Analysis

**Current State**: Simple scatter plot (predicted vs true)

**Improvements**:
- Confidence intervals (bootstrap)
- Prediction intervals (80%, 95%)
- Marginal distributions (rug plots)
- Density coloring (hexbin or KDE)
- Outlier highlighting

**New Function**: `plot_predictions_enhanced()`

```python
def plot_predictions_enhanced(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    percentiles: tuple = (5, 95),
    show_intervals: tuple = (68, 95),
    color_by: Optional[str] = 'density',
    highlight_outliers: bool = True,
    outlier_threshold: float = 2.0,
    language: str = 'both',
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
```

### 2.3 Residual Diagnostics

**New Function**: `plot_residual_diagnostics()`

**Contents**:
1. Residuals vs Predicted (check heteroscedasticity)
2. Q-Q plot (normality check)
3. Histogram of residuals with kernel density estimate
4. Scale-location plot (sqrt standardized residuals)

```python
def plot_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    language: str = 'both',
    figsize: tuple = (14, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
```

### 2.4 Model Comparison Dashboard

**New Function**: `plot_model_comparison()`

**Visualizations**:
- Radar chart for multiple metrics
- Box plots for cross-validation results
- Statistical comparison table
- Win/loss matrix

```python
def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['r2', 'mae', 'rmse', 'mape'],
    comparison_type: str = 'radar',
    language: str = 'both',
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
```

---

## 3. Feature Importance Visualizations

### 3.1 Enhanced Feature Importance

**Current State**: Simple horizontal bar chart

**Improvements**:
- Error bars (std across permutations)
- Significance markers (*, **, ***)
- Normalized percentages
- Color by feature type or significance

**New Function**: `plot_feature_importance_enhanced()`

```python
def plot_feature_importance_enhanced(
    importances: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    significance: Optional[np.ndarray] = None,
    normalize: bool = True,
    sort_by: str = 'importance',
    color_scheme: str = 'significance',
    language: str = 'both',
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
```

### 3.2 Timestep/Redshift Importance

**Current State**: Colored bar chart or line plot

**Improvements**:
- Heatmap visualization
- Confidence bands
- Theoretical model overlay
- Zoomed regions of interest

**New Function**: `plot_timestep_importance_enhanced()`

```python
def plot_timestep_importance_enhanced(
    importances: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    redshifts: Optional[np.ndarray] = None,
    show_confidence: bool = True,
    show_theory: bool = True,
    theory_model: Optional[str] = 'LCDM',
    highlight_regions: Optional[List[tuple]] = None,
    language: str = 'both',
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
```

### 3.3 Feature Correlation Matrix

**New Function**: `plot_feature_correlation()`

```python
def plot_feature_correlation(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    method: str = 'pearson',
    show_values: bool = True,
    language: str = 'both',
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
```

---

## 4. Cosmological Data Visualizations

### 4.1 H(z) Data Plots

**Current State**: Simple line plots of H(z) vs z

**Improvements**:
- Multiple samples with transparency
- Confidence bands (mean ± 1σ, 2σ)
- Theoretical model overlay
- Residuals as inset
- Observational data points with error bars

**New Function**: `plot_cosmological_data_enhanced()`

```python
def plot_cosmological_data_enhanced(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    y_std: Optional[np.ndarray] = None,
    n_samples: int = 20,
    show_samples: bool = True,
    show_mean: bool = True,
    show_confidence_bands: tuple = (1, 2),
    show_theory: bool = True,
    theory_params: Optional[Dict] = None,
    show_residuals: bool = True,
    residual_height: float = 0.15,
    observational_data: Optional[np.ndarray] = None,
    observational_errors: Optional[np.ndarray] = None,
    language: str = 'both',
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
```

### 4.2 Parameter Distribution Analysis

**New Function**: `plot_parameter_distributions()`

**Visualizations**:
- Violin plots for each parameter
- Corner plot (pairwise correlations)
- Marginal posteriors with credible intervals
- Parameter constraints table

```python
def plot_parameter_distributions(
    samples: Dict[str, np.ndarray],
    true_values: Optional[Dict[str, float]] = None,
    plot_type: str = 'violin',
    show_means: bool = True,
    show_medians: bool = True,
    show_credible_intervals: tuple = (68, 95),
    language: str = 'both',
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
```

### 4.3 Theory vs Observation Comparison

**New Function**: `plot_theory_observation_comparison()`

```python
def plot_theory_observation_comparison(
    z_obs: np.ndarray,
    h_obs: np.ndarray,
    h_err: Optional[np.ndarray] = None,
    z_theory: Optional[np.ndarray] = None,
    h_theory: Optional[np.ndarray] = None,
    theory_labels: Optional[List[str]] = None,
    cosmology_models: Optional[Dict] = None,
    residuals: bool = True,
    language: str = 'both',
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
```

### 4.4 Cosmological Constraints Plot

**New Function**: `plot_cosmological_constraints()`

For displaying constraints on (H₀, Ωₘ, w₀, wₐ) parameters

```python
def plot_cosmological_constraints(
    samples: Dict[str, np.ndarray],
    params: List[str] = ['h0', 'omega_m'],
    show_contours: bool = True,
    contour_levels: tuple = (0.393, 0.865, 0.989),
    show_means: bool = True,
    show_medians: bool = True,
    show_best_fit: bool = True,
    reference_values: Optional[Dict[str, float]] = None,
    language: str = 'both',
    figsize: tuple = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
```

---

## 5. Implementation Order

### Phase 1: Foundation (Week 1)
**Priority**: Critical

1. Create `utils/styles.py`
   - Define color palettes (Okabe-Ito, viridis)
   - Academic and presenter style dictionaries
   - Style application functions

2. Create `utils/labels.py`
   - BilingualLabel class
   - All label definitions
   - Axis formatting utilities

3. Update `utils/visualization.py`
   - Add `apply_style()` call to all functions
   - Add `language` parameter to existing functions
   - Add bilingual labels using new system

### Phase 2: Model Performance (Week 2)
**Priority**: High

1. Create `plot_training_history_enhanced()`
   - Smoothing algorithm
   - Confidence bands
   - Best epoch markers

2. Create `plot_predictions_enhanced()`
   - Confidence intervals
   - Density coloring
   - Outlier detection

3. Create `plot_residual_diagnostics()`
   - Q-Q plot
   - Scale-location plot
   - Normality tests

4. Create `plot_model_comparison()`
   - Radar chart
   - Box plots
   - Win matrix

### Phase 3: Feature Importance (Week 3)
**Priority**: Medium-High

1. Create `plot_feature_importance_enhanced()`
   - Error bars
   - Significance markers
   - Multiple color schemes

2. Create `plot_timestep_importance_enhanced()`
   - Heatmap view
   - Theory overlay
   - Zoom regions

3. Create `plot_feature_correlation()`
   - Correlation matrix heatmap
   - Clustering visualization

### Phase 4: Cosmological Data (Week 4)
**Priority**: Medium

1. Create `plot_cosmological_data_enhanced()`
   - Confidence bands
   - Theory comparison
   - Observational data overlay

2. Create `plot_parameter_distributions()`
   - Violin plots
   - Corner plots
   - Credible intervals

3. Create `plot_theory_observation_comparison()`
   - Residuals display
   - Multiple model comparison

4. Create `plot_cosmological_constraints()`
   - Contour plots
   - 1D posteriors

### Phase 5: Integration & Testing (Week 5-6)
**Priority**: Medium

1. Test all functions with real data
2. Update notebooks to use new functions
3. Generate publication-ready figures
4. Document all functions with examples
5. Add unit tests for visualization functions

---

## 6. Code Examples

### Example 1: Basic Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from utils.styles import apply_style
from utils.labels import get_label, format_axis
from utils.visualization import (
    plot_training_history_enhanced,
    plot_predictions_enhanced,
    plot_feature_importance_enhanced,
    plot_cosmological_data_enhanced
)

# Apply academic style
apply_style('academic')

# Load data
X, y = load_dataset('input/')
model = keras.models.load_model('models/cnn')

# Make predictions
y_pred = model.predict(X_test)
y_pred_std = np.std([model.predict(X_test_bootstrap) for _ in range(100)], axis=0)

# Plot predictions with confidence intervals
fig = plot_predictions_enhanced(
    y_true=y_test,
    y_pred=y_pred.flatten(),
    y_std=y_pred_std,
    show_intervals=(68, 95),
    color_by='density',
    language='both',
    save_path='imagens/predictions_enhanced.png'
)
plt.close(fig)
```

### Example 2: Model Comparison

```python
from utils.visualization import plot_model_comparison

results = {
    'CNN': {'r2': 0.95, 'mae': 2.1, 'rmse': 3.2},
    'RNN': {'r2': 0.93, 'mae': 2.5, 'rmse': 3.8},
    'Dense': {'r2': 0.91, 'mae': 2.8, 'rmse': 4.1},
    'SVM': {'r2': 0.89, 'mae': 3.1, 'rmse': 4.5},
}

fig = plot_model_comparison(
    results=results,
    metrics=['r2', 'mae', 'rmse'],
    comparison_type='radar',
    language='both'
)
plt.savefig('imagens/model_comparison_radar.png', dpi=300)
```

### Example 3: Cosmological Data with Theory

```python
from utils.visualization import plot_cosmological_data_enhanced
from cosmology.models import CosmologicalModel

# Load observational data (Bengaly et al.)
z_obs = np.array([0.07, 0.09, 0.12, ...])
h_obs = np.array([69.0, 69.0, 68.6, ...])
h_err = np.array([2.0, 2.0, 1.5, ...])

# Plot with theory comparison
fig = plot_cosmological_data_enhanced(
    X=X_test,
    y=y_test,
    n_samples=10,
    show_theory=True,
    theory_params={'H0': 67.45, 'Omega_m': 0.315},
    observational_data=z_obs,
    observational_errors=h_err,
    language='both',
    save_path='imagens/hz_cosmological.png'
)
```

---

## 7. Backward Compatibility

### Principles

1. **Additive Changes Only**: New functions are additions, not replacements
2. **Existing Code Works**: Current notebooks continue to function without modification
3. **Opt-In Enhancement**: New features require explicit adoption
4. **Gradual Migration**: Teams can migrate at their own pace

### Migration Strategy

**For New Code**:
```python
# Use new enhanced functions
from utils.visualization import plot_predictions_enhanced
fig = plot_predictions_enhanced(y_true, y_pred, language='both')
```

**For Existing Code**:
```python
# Continue using old functions (still supported)
from utils.visualization import plot_predictions  # Old function still works
fig = plot_predictions(y_true, y_pred)
```

**Gradual Upgrade**:
1. New notebooks use enhanced functions by default
2. Existing notebooks upgraded during natural workflow
3. Old functions deprecated after 6-month transition
4. Final removal after consensus from all users

### Deprecation Warnings

Old functions will emit deprecation warnings pointing to new alternatives:

```python
import warnings

def plot_predictions(y_true, y_pred, ...):
    warnings.warn(
        "plot_predictions is deprecated. Use plot_predictions_enhanced instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... original code
```

---

## Appendix A: Color Palette Reference

### Okabe-Ito Palette (Colorblind-Safe)

| Index | Color | Hex | Use Case |
|-------|-------|-----|----------|
| 0 | Blue | `#0072B2` | Primary data |
| 1 | Orange | `#E69F00` | Secondary data |
| 2 | Green | `#009E73` | Tertiary data |
| 3 | Pink | `#CC79A7` | Quaternary data |
| 4 | Sky Blue | `#56B4E9` | Accent 1 |
| 5 | Vermillion | `#D55E00` | Accent 2 |
| 6 | Yellow | `#F0E442` | Highlight |
| 7 | Black | `#000000` | Text/Lines |

### Sequential Palettes

- **viridis**: Default for sequential data (perceptually uniform)
- **plasma**: Alternative sequential (purple to yellow)
- **inferno**: High contrast sequential (black to yellow)
- **magma**: Bright sequential (black to pink)

---

## Appendix B: Figure Size Standards

### Academic Papers (Individual Venues)

**American Physical Society (APS)**:
- Single column: 3.375" (86mm)
- Double column: 6.75" (171mm)
- Full page: 8.5" (216mm)

**Elsevier**:
- Single column: 3.5" (89mm)
- 1.5 column: 5.5" (140mm)
- Double column: 7.3" (185mm)

**Nature**:
- Single column: 3.5" (89mm)
- Double column: 7.25" (184mm)
- Triple column: 11" (279mm)

### Thesis/Dissertation

- Generally: 6" × 4" (150mm × 100mm) for single plots
- Full page figures: 6" × 9" (150mm × 225mm)

---

## Appendix C: File Naming Convention

Use descriptive, consistent file names:

```
{type}_{metric/target}_{model}_{additional_info}.{ext}

Examples:
- loss_h0_cnn_training.png
- predictions_h0_rnn_bi_with_intervals.png
- feature_importance_omegam_svm.png
- timestep_importance_cnn_colored.png
- hz_data_lcdm_with_theory.png
- residuals_cnn_qq_plot.png
```

**Components**:
- `type`: loss, predictions, importance, hz_data, residuals
- `metric/target`: h0, omegam, w0, wa
- `model`: cnn, rnn, dense, svm, all
- `additional_info`: training, comparison, colored, with_intervals, etc.

---

## Appendix D: Testing Checklist

Before considering implementation complete:

- [ ] All functions have comprehensive docstrings
- [ ] All functions accept `language` parameter
- [ ] All functions apply academic style by default
- [ ] All functions have `save_path` and `show` parameters
- [ ] Bilingual labels render correctly in both modes
- [ ] Colorblind-safe palettes used throughout
- [ ] Figure sizes match academic standards
- [ ] File naming follows convention
- [ ] Unit tests cover core functionality
- [ ] Integration tests with real data pass
- [ ] Documentation updated with examples
- [ ] Deprecation path for old functions documented

---

## Appendix E: Related Files

**Existing Files**:
- `utils/visualization.py` - Current visualization functions
- `utils/data_loading.py` - Data loading utilities
- `SVM/svm_feature_importance.py` - SVM importance functions
- `feature_importance.ipynb` - Feature importance notebook

**New Files to Create**:
- `utils/styles.py` - Style configuration
- `utils/labels.py` - Bilingual labels
- `utils/plotting_enhanced.py` - Enhanced plotting functions

**Files to Update**:
- `utils/__init__.py` - Export new functions
- `AGENTS.md` - Document new visualization approach
- README.md - Link to visualization documentation

---

*End of GraphTODO Document*
