"""Data loading utilities for cosmological datasets."""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
from tqdm import tqdm
import os
import pickle
import hashlib


def load_dataset(
    data_dir: Union[str, Path],
    n_files: Optional[int] = None,
    file_pattern: str = "data_{}.npy",
    start_index: int = 1,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cosmological dataset from .npy files.
    
    Args:
        data_dir: Directory containing data files
        n_files: Number of files to load (loads all if None)
        file_pattern: Pattern for file names (use {} for index)
        start_index: Starting index for files
        verbose: Show progress bar
        
    Returns:
        Tuple of (X, y) where X has shape (n_samples, n_points, n_features)
        and y has shape (n_samples,)
    """
    data_dir = Path(data_dir)
    
    # Get list of files
    files = sorted(data_dir.glob(file_pattern.replace("{}", "*")))
    if n_files is not None:
        files = files[:n_files]
    
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {data_dir} with pattern {file_pattern}")
    
    if verbose:
        files = tqdm(files, desc="Loading data")
    
    X_list = []
    y_list = []
    
    for file in files:
        data = np.load(file)
        
        # Assuming data format: [z, H(z), n*H0, n*Omega_m, ...]
        # Extract features (z, H(z)) and target (H0)
        redshifts = data[:, 0]  # First column: z
        hubble = data[:, 1]     # Second column: H(z)
        h0_value = data[0, 2] / len(redshifts)  # Third column: n*H0
        
        # Create feature array with shape (n_points, 2) for (z, H(z))
        features = np.column_stack([redshifts, hubble])
        
        X_list.append(features)
        y_list.append(h0_value)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y


def load_dataset_parallel(
    data_dir: Union[str, Path],
    n_files: Optional[int] = None,
    file_pattern: str = "data_{}.npy",
    n_jobs: int = -1,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cosmological dataset in parallel using joblib.
    
    Args:
        data_dir: Directory containing data files
        n_files: Number of files to load (loads all if None)
        file_pattern: Pattern for file names (use {} for index)
        n_jobs: Number of parallel jobs (-1 = all cores)
        verbose: Show progress bar
        
    Returns:
        Tuple of (X, y)
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        print("joblib not installed. Falling back to sequential loading.")
        return load_dataset(data_dir, n_files, file_pattern, verbose=verbose)
    
    data_dir = Path(data_dir)
    
    # Get list of files
    files = sorted(data_dir.glob(file_pattern.replace("{}", "*")))
    if n_files is not None:
        files = files[:n_files]
    
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {data_dir} with pattern {file_pattern}")
    
    def load_single_file(file_path):
        """Load and process a single file."""
        data = np.load(file_path)
        
        redshifts = data[:, 0]
        hubble = data[:, 1]
        h0_value = data[0, 2] / len(redshifts)
        
        features = np.column_stack([redshifts, hubble])
        return features, h0_value
    
    if verbose:
        print(f"Loading {len(files)} files in parallel...")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(load_single_file)(file) for file in files
    )
    
    X_list, y_list = zip(*results)
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y


def load_dataset_cached(
    data_dir: Union[str, Path],
    cache_dir: Union[str, Path] = ".cache",
    n_files: Optional[int] = None,
    file_pattern: str = "data_{}.npy",
    force_reload: bool = False,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset with caching to speed up subsequent loads.
    
    Args:
        data_dir: Directory containing data files
        cache_dir: Directory to store cache files
        n_files: Number of files to load
        file_pattern: Pattern for file names
        force_reload: Force reload even if cache exists
        verbose: Show progress
        
    Returns:
        Tuple of (X, y)
    """
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    # Create cache key based on data directory and parameters
    cache_key = f"{data_dir}_{n_files}_{file_pattern}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = cache_dir / f"dataset_{cache_hash}.pkl"
    
    # Check if cache exists and is valid
    if cache_file.exists() and not force_reload:
        if verbose:
            print(f"Loading cached dataset from {cache_file}")
        
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Verify cache is still valid
        if cached_data['data_dir'] == str(data_dir) and \
           cached_data['n_files'] == n_files and \
           cached_data['file_pattern'] == file_pattern:
            return cached_data['X'], cached_data['y']
        else:
            if verbose:
                print("Cache invalid, reloading data...")
    
    # Load fresh data
    X, y = load_dataset(
        data_dir=data_dir,
        n_files=n_files,
        file_pattern=file_pattern,
        verbose=verbose
    )
    
    # Save to cache
    cache_data = {
        'X': X,
        'y': y,
        'data_dir': str(data_dir),
        'n_files': n_files,
        'file_pattern': file_pattern
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    if verbose:
        print(f"Dataset cached to {cache_file}")
    
    return X, y


def get_dataset_statistics(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Calculate basic statistics of the dataset.
    
    Args:
        X: Feature array (n_samples, n_points, n_features)
        y: Target array (n_samples,)
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'n_samples': len(X),
        'n_points': X.shape[1] if len(X.shape) > 1 else 1,
        'n_features': X.shape[2] if len(X.shape) > 2 else 2,
        'y_mean': np.mean(y),
        'y_std': np.std(y),
        'y_min': np.min(y),
        'y_max': np.max(y),
        'X_shape': X.shape,
        'y_shape': y.shape,
    }
    
    # Add statistics for each feature
    if len(X.shape) == 3:
        for feature_idx in range(X.shape[2]):
            feature_data = X[:, :, feature_idx].flatten()
            stats[f'feature_{feature_idx}_mean'] = np.mean(feature_data)
            stats[f'feature_{feature_idx}_std'] = np.std(feature_data)
            stats[f'feature_{feature_idx}_min'] = np.min(feature_data)
            stats[f'feature_{feature_idx}_max'] = np.max(feature_data)
    
    return stats


def normalize_dataset(
    X: np.ndarray, 
    y: np.ndarray,
    method: str = 'standard'
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Normalize dataset using various methods.
    
    Args:
        X: Feature array
        y: Target array
        method: Normalization method ('standard', 'minmax', 'robust')
        
    Returns:
        Tuple of (X_normalized, y_normalized, normalization_params)
    """
    normalization_params = {}
    
    if method == 'standard':
        # Z-score normalization
        X_mean = np.mean(X, axis=(0, 1), keepdims=True)
        X_std = np.std(X, axis=(0, 1), keepdims=True)
        y_mean = np.mean(y)
        y_std = np.std(y)
        
        X_norm = (X - X_mean) / (X_std + 1e-8)
        y_norm = (y - y_mean) / (y_std + 1e-8)
        
        normalization_params = {
            'method': 'standard',
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std
        }
        
    elif method == 'minmax':
        # Min-Max normalization to [0, 1]
        X_min = np.min(X, axis=(0, 1), keepdims=True)
        X_max = np.max(X, axis=(0, 1), keepdims=True)
        y_min = np.min(y)
        y_max = np.max(y)
        
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        y_norm = (y - y_min) / (y_max - y_min + 1e-8)
        
        normalization_params = {
            'method': 'minmax',
            'X_min': X_min,
            'X_max': X_max,
            'y_min': y_min,
            'y_max': y_max
        }
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return X_norm, y_norm, normalization_params