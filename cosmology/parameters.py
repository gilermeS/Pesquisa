"""Cosmological parameters from Planck 2018 and other sources."""
from dataclasses import dataclass
from typing import Optional


# Planck 2018 parameters (flat LCDM)
OMEGA_M = 0.315
SIGMA_OMEGA_M = 0.007
OMEGA_CH2 = 0.120
SIGMA_OMEGA_CH2 = 0.001
OMEGA_BH2 = 0.0224
SIGMA_OMEGA_BH2 = 0.0001
H0 = 67.45
SIGMA_H0 = 0.62

# wCDM parameters
W0 = -1.0
SIGMA_W0 = 0.0  # No uncertainty by default

# w(a)CDM parameters (CPL parameterization)
WA = 0.0
SIGMA_WA = 0.0  # No uncertainty by default


@dataclass
class CosmologicalParameters:
    """
    Container for cosmological parameters.
    
    Attributes:
        omega_m: Matter density parameter (Ω_m)
        sigma_omega_m: Uncertainty in Ω_m
        omega_ch2: Physical cold dark matter density (Ω_c h²)
        sigma_omega_ch2: Uncertainty in Ω_c h²
        omega_bh2: Physical baryon density (Ω_b h²)
        sigma_omega_bh2: Uncertainty in Ω_b h²
        h0: Hubble constant in units of 100 km/s/Mpc (h)
        sigma_h0: Uncertainty in h
        w0: Dark energy equation of state parameter at z=0
        sigma_w0: Uncertainty in w0
        wa: Dark energy evolution parameter
        sigma_wa: Uncertainty in wa
    """
    omega_m: float = OMEGA_M
    sigma_omega_m: float = SIGMA_OMEGA_M
    omega_ch2: float = OMEGA_CH2
    sigma_omega_ch2: float = SIGMA_OMEGA_CH2
    omega_bh2: float = OMEGA_BH2
    sigma_omega_bh2: float = SIGMA_OMEGA_BH2
    h0: float = H0
    sigma_h0: float = SIGMA_H0
    w0: float = W0
    sigma_w0: float = SIGMA_W0
    wa: float = WA
    sigma_wa: float = SIGMA_WA
    
    @classmethod
    def lcdm(cls) -> 'CosmologicalParameters':
        """Create LCDM parameters (w0 = -1, wa = 0)."""
        return cls(w0=-1.0, wa=0.0, sigma_w0=0.0, sigma_wa=0.0)
    
    @classmethod
    def wcdm(cls, w0: float = -1.0) -> 'CosmologicalParameters':
        """Create wCDM parameters (constant w, wa = 0)."""
        return cls(w0=w0, wa=0.0, sigma_wa=0.0)
    
    @classmethod
    def wacdm(cls, w0: float = -0.9, wa: float = -0.5) -> 'CosmologicalParameters':
        """Create w(a)CDM parameters (CPL parameterization)."""
        return cls(w0=w0, wa=wa)
    
    def to_dict(self) -> dict:
        """Convert parameters to dictionary."""
        return {
            'omega_m': self.omega_m,
            'sigma_omega_m': self.sigma_omega_m,
            'omega_ch2': self.omega_ch2,
            'sigma_omega_ch2': self.sigma_omega_ch2,
            'omega_bh2': self.omega_bh2,
            'sigma_omega_bh2': self.sigma_omega_bh2,
            'h0': self.h0,
            'sigma_h0': self.sigma_h0,
            'w0': self.w0,
            'sigma_w0': self.sigma_w0,
            'wa': self.wa,
            'sigma_wa': self.sigma_wa,
        }
    
    @classmethod
    def from_dict(cls, params_dict: dict) -> 'CosmologicalParameters':
        """Create parameters from dictionary."""
        return cls(**params_dict)