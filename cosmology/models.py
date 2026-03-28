"""Cosmological models for calculating Hubble parameter."""
import numpy as np
from typing import Union, List, Optional
from .parameters import CosmologicalParameters


def friedmann_equation(
    redshift: Union[float, np.ndarray],
    hubble_parameter: float,
    omega_matter: float
) -> Union[float, np.ndarray]:
    """
    Calculate Hubble parameter using Friedmann equation for flat LCDM model.
    
    H(z) = H0 * sqrt(Ω_m * (1+z)^3 + (1-Ω_m))
    
    Args:
        redshift: Cosmological redshift (z) - can be scalar or array
        hubble_parameter: Hubble parameter H0 in units of km/s/Mpc
        omega_matter: Matter density parameter Ω_m
        
    Returns:
        Hubble parameter at given redshift(s) in km/s/Mpc
    """
    return hubble_parameter * np.sqrt(
        omega_matter * (1 + redshift)**3.0 + (1.0 - omega_matter)
    )


def friedmann_wcdm_equation(
    redshift: Union[float, np.ndarray],
    hubble_parameter: float,
    omega_matter: float,
    w0: float = -1.0
) -> Union[float, np.ndarray]:
    """
    Calculate Hubble parameter for wCDM model (constant dark energy EoS).
    
    H(z) = H0 * sqrt(Ω_m * (1+z)^3 + (1-Ω_m) * (1+z)^(3(1+w0)))
    
    Args:
        redshift: Cosmological redshift (z)
        hubble_parameter: Hubble parameter H0 in km/s/Mpc
        omega_matter: Matter density parameter Ω_m
        w0: Dark energy equation of state parameter (constant)
        
    Returns:
        Hubble parameter at given redshift(s) in km/s/Mpc
    """
    return hubble_parameter * np.sqrt(
        omega_matter * (1 + redshift)**3.0 + 
        (1.0 - omega_matter) * (1 + redshift)**(3.0 * (1.0 + w0))
    )


def friedmann_wacdm_equation(
    redshift: Union[float, np.ndarray],
    hubble_parameter: float,
    omega_matter: float,
    w0: float = -0.9,
    wa: float = -0.5
) -> Union[float, np.ndarray]:
    """
    Calculate Hubble parameter for w(a)CDM model (CPL parameterization).
    
    H(z) = H0 * sqrt(Ω_m * (1+z)^3 + (1-Ω_m) * (1+z)^(3(1+w0+wa)) * exp(-3*wa*z/(1+z)))
    
    Args:
        redshift: Cosmological redshift (z)
        hubble_parameter: Hubble parameter H0 in km/s/Mpc
        omega_matter: Matter density parameter Ω_m
        w0: Dark energy equation of state parameter at z=0
        wa: Dark energy evolution parameter
        
    Returns:
        Hubble parameter at given redshift(s) in km/s/Mpc
    """
    ez = (
        omega_matter * (1 + redshift)**3.0 + 
        (1.0 - omega_matter) * (1 + redshift)**(3.0 * (1.0 + w0 + wa)) * 
        np.exp(-3.0 * wa * redshift / (1.0 + redshift))
    )
    return hubble_parameter * np.sqrt(ez)


class CosmologicalModel:
    """Wrapper class for cosmological models with parameter management."""
    
    def __init__(self, model_type: str = 'LCDM', parameters: Optional[CosmologicalParameters] = None):
        """
        Initialize cosmological model.
        
        Args:
            model_type: One of 'LCDM', 'wCDM', or 'wACDM'
            parameters: Cosmological parameters (uses defaults if None)
        """
        self.model_type = model_type.upper()
        if parameters is None:
            if self.model_type == 'LCDM':
                self.parameters = CosmologicalParameters.lcdm()
            elif self.model_type == 'WCDM':
                self.parameters = CosmologicalParameters.wcdm()
            elif self.model_type == 'WACDM':
                self.parameters = CosmologicalParameters.wacdm()
            else:
                raise ValueError(f"Unknown model type: {model_type}. Use 'LCDM', 'wCDM', or 'wACDM'.")
        else:
            self.parameters = parameters
    
    def hubble_parameter(
        self, 
        redshift: Union[float, np.ndarray],
        h0: Optional[float] = None,
        omega_m: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate Hubble parameter H(z) for this model.
        
        Args:
            redshift: Redshift(s) to evaluate at
            h0: Override Hubble constant (optional)
            omega_m: Override matter density (optional)
            
        Returns:
            H(z) in km/s/Mpc
        """
        h0 = h0 if h0 is not None else self.parameters.h0
        omega_m = omega_m if omega_m is not None else self.parameters.omega_m
        
        if self.model_type == 'LCDM':
            return friedmann_equation(redshift, h0, omega_m)
        elif self.model_type == 'WCDM':
            return friedmann_wcdm_equation(redshift, h0, omega_m, self.parameters.w0)
        elif self.model_type == 'WACDM':
            return friedmann_wacdm_equation(
                redshift, h0, omega_m, self.parameters.w0, self.parameters.wa
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def __repr__(self) -> str:
        return f"CosmologicalModel(model_type='{self.model_type}', parameters={self.parameters})"