"""Data generation script for cosmological simulations (LCDM model, 47 points)."""
import sys
import os

# Add parent directory to path to import cosmology package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cosmology.generators import generate_lcdm_data


def main():
    """Generate LCDM simulations with 47 redshift points."""
    print("Generating LCDM simulations with 47 redshift points...")
    generate_lcdm_data(
        n_simulations=10000,
        n_redshift_points=47,
        redshift_min=0.1,
        redshift_max=1.5,
        output_dir='input47/'
    )
    print("Done!")


if __name__ == "__main__":
    main()