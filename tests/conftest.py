"""Pytest configuration and fixtures."""
import pytest
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def sample_data():
    """Create sample cosmological data for testing."""
    # Create sample data similar to real cosmological data
    n_samples = 50
    n_points = 10
    
    # Redshift array
    redshifts = np.linspace(0.1, 1.5, n_points)
    
    # H(z) values (LCDM with H0=70, Omega_m=0.3)
    h0_true = 70.0
    omega_m_true = 0.3
    hubble_values = h0_true * np.sqrt(omega_m_true * (1 + redshifts)**3 + (1 - omega_m_true))
    
    # Create dataset
    X = []
    y = []
    
    for i in range(n_samples):
        # Add small random variations
        h0_variation = np.random.normal(0, 2.0)  # ±2 km/s/Mpc variation
        h0_sample = h0_true + h0_variation
        
        # Calculate H(z) with this H0
        hz_sample = h0_sample * np.sqrt(omega_m_true * (1 + redshifts)**3 + (1 - omega_m_true))
        
        # Add small noise
        noise = np.random.normal(0, 0.5, n_points)
        hz_sample += noise
        
        # Features: (z, H(z))
        features = np.column_stack([redshifts, hz_sample])
        
        X.append(features)
        y.append(h0_sample)
    
    return np.array(X), np.array(y)


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def lcdm_model():
    """Create LCDM model for testing."""
    from cosmology.models import CosmologicalModel
    return CosmologicalModel('LCDM')


@pytest.fixture
def wcdm_model():
    """Create wCDM model for testing."""
    from cosmology.models import CosmologicalModel
    return CosmologicalModel('wCDM')


@pytest.fixture
def wacdm_model():
    """Create wACDM model for testing."""
    from cosmology.models import CosmologicalModel
    return CosmologicalModel('wACDM')


@pytest.fixture(params=['LCDM', 'wCDM', 'wACDM'])
def all_models(request):
    """Parameterized fixture for all cosmological models."""
    from cosmology.models import CosmologicalModel
    return CosmologicalModel(request.param)


@pytest.fixture
def sample_parameters():
    """Create sample cosmological parameters."""
    from cosmology.parameters import CosmologicalParameters
    return CosmologicalParameters(
        omega_m=0.315,
        sigma_omega_m=0.007,
        h0=67.45,
        sigma_h0=0.62,
        w0=-1.0,
        wa=0.0
    )


@pytest.fixture
def trained_model():
    """Create a simple trained model for testing."""
    import tensorflow as tf
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(80, 2)),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Create dummy data for training
    x_train = np.random.randn(100, 80, 2)
    y_train = np.random.randn(100)
    
    # Train for 1 epoch
    model.fit(x_train, y_train, epochs=1, verbose=0)
    
    return model


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Skip slow tests by default
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )