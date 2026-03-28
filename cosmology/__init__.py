"""Cosmology package for generating and analyzing cosmological data."""
from .models import (
    friedmann_equation,
    friedmann_wacdm_equation,
    CosmologicalModel,
)
from .parameters import (
    OMEGA_M,
    OMEGA_CH2,
    OMEGA_BH2,
    H0,
    W0,
    WA,
    CosmologicalParameters,
)
from .generators import (
    generate_parameter_sample,
    MonteCarloGenerator,
    generate_lcdm_data,
    generate_wcdm_data,
    generate_wacdm_data,
)

__version__ = "1.0.0"
__author__ = "Guilherme de Souza Ramos Cardoso"