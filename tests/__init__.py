"""Test suite for cosmology deep learning project."""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_DATA_DIR = project_root / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)