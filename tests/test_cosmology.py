"""Tests for cosmology package."""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from cosmology.models import (
    friedmann_equation,
    friedmann_wcdm_equation,
    friedmann_wacdm_equation,
    CosmologicalModel
)
from cosmology.parameters import CosmologicalParameters
from cosmology.generators import (
    generate_parameter_sample,
    MonteCarloGenerator,
    generate_lcdm_data,
    generate_wcdm_data,
    generate_wacdm_data
)


class TestFriedmannEquations:
    """Test Friedmann equation implementations."""
    
    def test_friedmann_equation_lcdm(self):
        """Test LCDM Friedmann equation."""
        z = np.array([0.0, 0.5, 1.0])
        h0 = 70.0
        omega_m = 0.3
        
        result = friedmann_equation(z, h0, omega_m)
        
        # Check shape
        assert result.shape == z.shape
        
        # Check physical constraints
        assert np.all(result > 0)
        
        # Check specific values
        expected_at_z0 = h0  # H(0) = H0
        assert np.isclose(result[0], expected_at_z0, rtol=1e-10)
        
        # H(z) should increase with z for LCDM
        assert np.all(np.diff(result) > 0)
    
    def test_friedmann_equation_scalar_input(self):
        """Test with scalar input."""
        z = 0.5
        h0 = 70.0
        omega_m = 0.3
        
        result = friedmann_equation(z, h0, omega_m)
        
        assert isinstance(result, (float, np.floating))
        assert result > 0
    
    def test_friedmann_wcdm_equation(self):
        """Test wCDM Friedmann equation."""
        z = np.array([0.0, 1.0])
        h0 = 70.0
        omega_m = 0.3
        w0 = -1.2
        
        result = friedmann_wcdm_equation(z, h0, omega_m, w0)
        
        assert result.shape == z.shape
        assert np.all(result > 0)
        
        # For w0 = -1, should match LCDM
        result_lcdm = friedmann_wcdm_equation(z, h0, omega_m, -1.0)
        result_lcdm_direct = friedmann_equation(z, h0, omega_m)
        np.testing.assert_allclose(result_lcdm, result_lcdm_direct, rtol=1e-10)
    
    def test_friedmann_wacdm_equation(self):
        """Test wACDM Friedmann equation."""
        z = np.array([0.0, 0.5, 1.0])
        h0 = 70.0
        omega_m = 0.3
        w0 = -0.9
        wa = -0.5
        
        result = friedmann_wacdm_equation(z, h0, omega_m, w0, wa)
        
        assert result.shape == z.shape
        assert np.all(result > 0)
        
        # For w0 = -1, wa = 0, should match LCDM
        result_lcdm = friedmann_wacdm_equation(z, h0, omega_m, -1.0, 0.0)
        result_lcdm_direct = friedmann_equation(z, h0, omega_m)
        np.testing.assert_allclose(result_lcdm, result_lcdm_direct, rtol=1e-10)


class TestCosmologicalParameters:
    """Test cosmological parameters."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = CosmologicalParameters()
        
        assert params.omega_m == 0.315
        assert params.h0 == 67.45
        assert params.w0 == -1.0
        assert params.wa == 0.0
    
    def test_lcdm_factory(self):
        """Test LCDM parameter factory."""
        params = CosmologicalParameters.lcdm()
        
        assert params.w0 == -1.0
        assert params.wa == 0.0
        assert params.sigma_w0 == 0.0
        assert params.sigma_wa == 0.0
    
    def test_wcdm_factory(self):
        """Test wCDM parameter factory."""
        params = CosmologicalParameters.wcdm(w0=-1.2)
        
        assert params.w0 == -1.2
        assert params.wa == 0.0
        assert params.sigma_wa == 0.0
    
    def test_wacdm_factory(self):
        """Test wACDM parameter factory."""
        params = CosmologicalParameters.wacdm(w0=-0.9, wa=-0.5)
        
        assert params.w0 == -0.9
        assert params.wa == -0.5
    
    def test_parameter_serialization(self):
        """Test parameter serialization/deserialization."""
        params = CosmologicalParameters(
            omega_m=0.3,
            h0=70.0,
            w0=-1.1,
            wa=-0.2
        )
        
        # Convert to dict and back
        params_dict = params.to_dict()
        params_restored = CosmologicalParameters.from_dict(params_dict)
        
        # Check all values match
        assert params_restored.omega_m == params.omega_m
        assert params_restored.h0 == params.h0
        assert params_restored.w0 == params.w0
        assert params_restored.wa == params.wa


class TestCosmologicalModel:
    """Test CosmologicalModel class."""
    
    def test_lcdm_model(self):
        """Test LCDM model."""
        model = CosmologicalModel('LCDM')
        
        z = np.array([0.0, 0.5, 1.0])
        h_z = model.hubble_parameter(z)
        
        assert h_z.shape == z.shape
        assert np.all(h_z > 0)
        assert np.isclose(h_z[0], model.parameters.h0, rtol=1e-10)
    
    def test_wcdm_model(self):
        """Test wCDM model."""
        model = CosmologicalModel('WCDM')
        
        z = np.array([0.0, 1.0])
        h_z = model.hubble_parameter(z)
        
        assert h_z.shape == z.shape
        assert np.all(h_z > 0)
    
    def test_wacdm_model(self):
        """Test wACDM model."""
        model = CosmologicalModel('WACDM')
        
        z = np.array([0.0, 0.5, 1.0])
        h_z = model.hubble_parameter(z)
        
        assert h_z.shape == z.shape
        assert np.all(h_z > 0)
    
    def test_model_with_custom_parameters(self):
        """Test model with custom parameters."""
        params = CosmologicalParameters(h0=80.0, omega_m=0.4)
        model = CosmologicalModel('LCDM', parameters=params)
        
        assert model.parameters.h0 == 80.0
        assert model.parameters.omega_m == 0.4
        
        z = np.array([0.0, 1.0])
        h_z = model.hubble_parameter(z)
        
        # H(z=0) should be H0
        assert np.isclose(h_z[0], 80.0, rtol=1e-10)


class TestGenerators:
    """Test data generation functions."""
    
    def test_generate_parameter_sample(self):
        """Test parameter sampling."""
        center = 100.0
        sigma = 5.0
        
        # Generate many samples
        samples = [generate_parameter_sample(center, sigma) for _ in range(1000)]
        
        # Check they're within bounds (3-sigma)
        assert all(center - 3 * sigma <= s <= center + 3 * sigma for s in samples)
        
        # Check approximate mean
        mean = np.mean(samples)
        assert abs(mean - center) < 1.0  # Should be close to center
    
    def test_monte_carlo_generator_lcdm(self):
        """Test Monte Carlo generator for LCDM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = MonteCarloGenerator(
                model_type='LCDM',
                n_simulations=10,
                n_redshift_points=5,
                output_dir=tmpdir
            )
            
            # Generate single simulation
            data, params = generator.generate_single_simulation(0)
            
            # Check data shape
            assert data.shape == (5, 4)  # [z, H(z), n*H0, n*Omega_m]
            
            # Check parameters
            assert 'h0' in params
            assert 'omega_m' in params
            assert 60 < params['h0'] < 80  # Within sampling range
            assert 0.2 < params['omega_m'] < 0.5
            
            # Check file was created
            assert (Path(tmpdir) / 'data_1.npy').exists()
    
    def test_generate_lcdm_data(self):
        """Test LCDM data generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = generate_lcdm_data(
                n_simulations=5,
                n_redshift_points=10,
                output_dir=tmpdir,
                verbose=False
            )
            
            # Check files were created
            assert (Path(tmpdir) / 'data_1.npy').exists()
            assert (Path(tmpdir) / 'data_real10.npy').exists()
            
            # Load and check data
            data = np.load(Path(tmpdir) / 'data_1.npy')
            assert data.shape == (10, 4)
    
    def test_generate_wcdm_data(self):
        """Test wCDM data generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = generate_wcdm_data(
                w0=-1.2,
                n_simulations=3,
                n_redshift_points=5,
                output_dir=tmpdir,
                verbose=False
            )
            
            assert (Path(tmpdir) / 'data_1.npy').exists()
    
    def test_generate_wacdm_data(self):
        """Test wACDM data generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = generate_wacdm_data(
                w0=-0.9,
                wa=-0.5,
                n_simulations=3,
                n_redshift_points=5,
                output_dir=tmpdir,
                verbose=False
            )
            
            assert (Path(tmpdir) / 'data_1.npy').exists()


class TestDataLoadingUtils:
    """Test data loading utilities."""
    
    def test_load_dataset(self):
        """Test dataset loading."""
        # Create temporary test data
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test .npy files
            for i in range(3):
                data = np.array([
                    [0.1, 70.0, 80 * 67.45, 80 * 0.315],
                    [0.2, 72.0, 80 * 67.45, 80 * 0.315],
                    [0.3, 74.0, 80 * 67.45, 80 * 0.315]
                ])
                np.save(Path(tmpdir) / f'data_{i+1}.npy', data)
            
            # Import here to avoid import errors during collection
            from utils.data_loading import load_dataset
            
            X, y = load_dataset(tmpdir, verbose=False)
            
            # Check shapes
            assert X.shape == (3, 3, 2)  # 3 samples, 3 points, 2 features
            assert y.shape == (3,)
            
            # Check values
            assert np.all(y > 60)  # H0 values should be reasonable
    
    def test_normalize_dataset(self):
        """Test dataset normalization."""
        from utils.data_loading import normalize_dataset
        
        # Create test data
        X = np.random.randn(10, 5, 2)
        y = np.random.randn(10) * 10 + 70
        
        # Test standard normalization
        X_norm, y_norm, params = normalize_dataset(X, y, method='standard')
        
        # Check normalization
        assert abs(np.mean(X_norm)) < 0.1  # Should be close to zero
        assert abs(np.std(X_norm) - 1.0) < 0.1  # Should be close to 1
        assert abs(np.mean(y_norm)) < 0.1
        assert abs(np.std(y_norm) - 1.0) < 0.1
        
        # Check params
        assert 'X_mean' in params
        assert 'X_std' in params
        assert 'y_mean' in params
        assert 'y_std' in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])