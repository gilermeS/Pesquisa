"""Style configuration for academic visualizations."""
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any

OKABE_ITO = [
    '#0072B2',
    '#E69F00',
    '#009E73',
    '#CC79A7',
    '#56B4E9',
    '#D55E00',
    '#F0E442',
    '#000000',
]

COLORBLIND_SAFE = OKABE_ITO

SEQUENTIAL_PALETTES = {
    'viridis': 'viridis',
    'plasma': 'plasma',
    'inferno': 'inferno',
    'magma': 'magma',
}

FIGURE_SIZES = {
    'single': (3.5, 2.625),
    'double': (7.0, 5.25),
    'full': (7.0, 5.25),
    'square': (5.0, 5.0),
    'wide': (10.0, 4.0),
    'tall': (5.0, 7.0),
}

ACADEMIC_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.title_fontsize': 10,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'errorbar.capsize': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.format': 'png',
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
    """Apply a predefined style to matplotlib."""
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


def _apply_dark_style() -> None:
    """Apply dark theme for presentations."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.grid': True,
        'grid.alpha': 0.2,
        'figure.facecolor': 'black',
        'axes.facecolor': '#1a1a1a',
    })


def get_color_cycle(n_colors: int = 8) -> list:
    """Get color cycle for plots."""
    return OKABE_ITO[:n_colors]
