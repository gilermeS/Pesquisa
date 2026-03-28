"""Data generation utilities for cosmological simulations."""
import numpy as np
import random
from typing import List, Tuple, Optional, Union
from pathlib import Path
from tqdm import tqdm

from .parameters import CosmologicalParameters
from .models import CosmologicalModel


def generate_parameter_sample(center: float, sigma: float, n_sigma: float = 3.0) -> float:
    """
    Generate a random parameter value from a Gaussian distribution.
    
    Args:
        center: Central value of the distribution
        sigma: Standard deviation
        n_sigma: Number of standard deviations for bounds (default: 3)
        
    Returns:
        Random value sampled from uniform distribution within n_sigma bounds
    """
    return random.uniform(center - n_sigma * sigma, center + n_sigma * sigma)


class MonteCarloGenerator:
    """
    Monte Carlo generator for cosmological data.
    
    Generates synthetic cosmological data by sampling cosmological parameters
    from Gaussian distributions and calculating H(z) for each realization.
    """
    
    def __init__(
        self,
        model_type: str = 'LCDM',
        parameters: Optional[CosmologicalParameters] = None,
        n_simulations: int = 10000,
        n_redshift_points: int = 80,
        redshift_min: float = 0.1,
        redshift_max: float = 1.5,
        output_dir: Union[str, Path] = 'input/'
    ):
        """
        Initialize Monte Carlo generator.
        
        Args:
            model_type: Cosmological model ('LCDM', 'wCDM', 'wACDM')
            parameters: Base cosmological parameters
            n_simulations: Number of simulations to generate
            n_redshift_points: Number of redshift points per simulation
            redshift_min: Minimum redshift value
            redshift_max: Maximum redshift value
            output_dir: Directory to save output .npy files
        """
        self.model_type = model_type.upper()
        self.parameters = parameters or CosmologicalParameters.lcdm()
        self.n_simulations = n_simulations
        self.n_redshift_points = n_redshift_points
        self.redshift_min = redshift_min
        self.redshift_max = redshift_max
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cosmological model
        self.model = CosmologicalModel(model_type, self.parameters)
    
    def generate_redshift_array(self) -> np.ndarray:
        """Generate linearly spaced redshift array."""
        return self.redshift_min + (self.redshift_max - self.redshift_min) * \
               np.arange(self.n_redshift_points) / (self.n_redshift_points - 1.0)
    
    def sample_parameters(self) -> Tuple[float, float, Optional[float], Optional[float]]:
        """
        Sample cosmological parameters from Gaussian distributions.
        
        Returns:
            Tuple of (H0, Omega_m, w0, wa) where w0, wa are None for LCDM
        """
        # Sample Hubble constant
        h0 = generate_parameter_sample(self.parameters.h0, self.parameters.sigma_h0, n_sigma=5.0)
        
        # Sample matter density
        omega_m = generate_parameter_sample(self.parameters.omega_m, self.parameters.sigma_omega_m)
        
        w0 = None
        wa = None
        
        if self.model_type == 'WCDM':
            w0 = generate_parameter_sample(self.parameters.w0, self.parameters.sigma_w0, n_sigma=2.0)
        elif self.model_type == 'WACDM':
            w0 = generate_parameter_sample(self.parameters.w0, self.parameters.sigma_w0, n_sigma=2.0)
            wa = generate_parameter_sample(self.parameters.wa, self.parameters.sigma_wa, n_sigma=2.0)
        
        return h0, omega_m, w0, wa
    
    def calculate_hubble_parameter(
        self, 
        redshifts: np.ndarray,
        h0: float,
        omega_m: float,
        w0: Optional[float] = None,
        wa: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate H(z) for given parameters.
        
        Args:
            redshifts: Array of redshift values
            h0: Hubble constant
            omega_m: Matter density parameter
            w0: Dark energy equation of state (for wCDM/wACDM)
            wa: Dark energy evolution parameter (for wACDM)
            
        Returns:
            Array of H(z) values
        """
        return self.model.hubble_parameter(redshifts, h0, omega_m)
    
    def generate_single_simulation(self, simulation_index: int) -> Tuple[np.ndarray, dict]:
        """
        Generate a single cosmological simulation.
        
        Args:
            simulation_index: Index of the simulation (for file naming)
            
        Returns:
            Tuple of (data_array, parameters_dict)
        """
        redshifts = self.generate_redshift_array()
        h0, omega_m, w0, wa = self.sample_parameters()
        
        # Calculate H(z)
        hubble_array = self.calculate_hubble_parameter(redshifts, h0, omega_m, w0, wa)
        
        # Prepare data for saving
        if self.model_type == 'LCDM':
            data_to_save = np.transpose([
                redshifts,
                hubble_array,
                self.n_redshift_points * np.array([h0]),
                self.n_redshift_points * np.array([omega_m])
            ])
        elif self.model_type == 'WCDM':
            data_to_save = np.transpose([
                redshifts,
                hubble_array,
                self.n_redshift_points * np.array([h0]),
                self.n_redshift_points * np.array([omega_m]),
                self.n_redshift_points * np.array([w0])
            ])
        elif self.model_type == 'WACDM':
            data_to_save = np.transpose([
                redshifts,
                hubble_array,
                self.n_redshift_points * np.array([h0]),
                self.n_redshift_points * np.array([omega_m]),
                self.n_redshift_points * np.array([w0]),
                self.n_redshift_points * np.array([wa])
            ])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Save to file
        filename = self.output_dir / f'data_{simulation_index + 1}'
        np.save(filename, data_to_save)
        
        # Return parameters for logging
        params_dict = {'h0': h0, 'omega_m': omega_m}
        if w0 is not None:
            params_dict['w0'] = w0
        if wa is not None:
            params_dict['wa'] = wa
            
        return data_to_save, params_dict
    
    def generate_all_simulations(self, verbose: bool = True) -> None:
        """
        Generate all simulations.
        
        Args:
            verbose: Show progress bar
        """
        if verbose:
            iterator = tqdm(range(self.n_simulations), desc="Generating simulations")
        else:
            iterator = range(self.n_simulations)
        
        for i in iterator:
            self.generate_single_simulation(i)
        
        # Generate "real" data with fiducial parameters
        self.generate_real_data()
    
    def generate_real_data(self) -> None:
        """Generate "real" data with fiducial cosmological parameters."""
        redshifts = self.generate_redshift_array()
        hubble_real = self.model.hubble_parameter(redshifts)
        
        if self.model_type == 'LCDM':
            real_data = np.transpose([redshifts, hubble_real])
        elif self.model_type == 'WCDM':
            real_data = np.transpose([redshifts, hubble_real])
        elif self.model_type == 'WACDM':
            real_data = np.transpose([redshifts, hubble_real])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        filename = self.output_dir / f'data_real{self.n_redshift_points}'
        np.save(filename, real_data)


def generate_lcdm_data(
    n_simulations: int = 10000,
    n_redshift_points: int = 80,
    redshift_min: float = 0.1,
    redshift_max: float = 1.5,
    output_dir: Union[str, Path] = 'input/',
    verbose: bool = True
) -> MonteCarloGenerator:
    """
    Convenience function to generate LCDM data.
    
    Args:
        n_simulations: Number of Monte Carlo simulations
        n_redshift_points: Number of redshift points per simulation
        redshift_min: Minimum redshift value
        redshift_max: Maximum redshift value
        output_dir: Directory to save output .npy files
        verbose: Show progress
        
    Returns:
        MonteCarloGenerator instance used for generation
    """
    generator = MonteCarloGenerator(
        model_type='LCDM',
        n_simulations=n_simulations,
        n_redshift_points=n_redshift_points,
        redshift_min=redshift_min,
        redshift_max=redshift_max,
        output_dir=output_dir
    )
    generator.generate_all_simulations(verbose=verbose)
    return generator


def generate_wcdm_data(
    w0: float = -1.0,
    n_simulations: int = 10000,
    n_redshift_points: int = 20,
    redshift_min: float = 0.1,
    redshift_max: float = 1.5,
    output_dir: Union[str, Path] = 'input/',
    verbose: bool = True
) -> MonteCarloGenerator:
    """
    Convenience function to generate wCDM data.
    
    Args:
        w0: Dark energy equation of state parameter
        n_simulations: Number of Monte Carlo simulations
        n_redshift_points: Number of redshift points per simulation
        redshift_min: Minimum redshift value
        redshift_max: Maximum redshift value
        output_dir: Directory to save output .npy files
        verbose: Show progress
        
    Returns:
        MonteCarloGenerator instance used for generation
    """
    params = CosmologicalParameters.wcdm(w0=w0)
    generator = MonteCarloGenerator(
        model_type='wCDM',
        parameters=params,
        n_simulations=n_simulations,
        n_redshift_points=n_redshift_points,
        redshift_min=redshift_min,
        redshift_max=redshift_max,
        output_dir=output_dir
    )
    generator.generate_all_simulations(verbose=verbose)
    return generator


def generate_wacdm_data(
    w0: float = -0.9,
    wa: float = -0.5,
    n_simulations: int = 10000,
    n_redshift_points: int = 20,
    redshift_min: float = 0.1,
    redshift_max: float = 1.5,
    output_dir: Union[str, Path] = 'input/',
    verbose: bool = True
) -> MonteCarloGenerator:
    """
    Convenience function to generate wACDM data.
    
    Args:
        w0: Dark energy equation of state parameter at z=0
        wa: Dark energy evolution parameter
        n_simulations: Number of Monte Carlo simulations
        n_redshift_points: Number of redshift points per simulation
        redshift_min: Minimum redshift value
        redshift_max: Maximum redshift value
        output_dir: Directory to save output .npy files
        verbose: Show progress
        
    Returns:
        MonteCarloGenerator instance used for generation
    """
    params = CosmologicalParameters.wacdm(w0=w0, wa=wa)
    generator = MonteCarloGenerator(
        model_type='wACDM',
        parameters=params,
        n_simulations=n_simulations,
        n_redshift_points=n_redshift_points,
        redshift_min=redshift_min,
        redshift_max=redshift_max,
        output_dir=output_dir
    )
    generator.generate_all_simulations(verbose=verbose)
    return generator