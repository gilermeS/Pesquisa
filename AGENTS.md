# AGENTS.md - Guidelines for Coding Agents

This document provides instructions for AI coding agents (e.g., Copilot, Cursor) working on this repository.

## Build & Run

### Environment Setup
1. Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies (if not already):
   ```powershell
   pip install tensorflow scikit-learn numpy pandas matplotlib tqdm joblib black isort flake8 mypy
   ```

### Project Structure
```
Pesquisa/
├── cosmology/           # Cosmological models and data generation
├── utils/               # Utility functions for data loading and visualization
├── Geradores/           # LCDM data generation scripts
├── SVM/                 # SVM feature importance analysis
├── CNN/, Dense/, RNN/   # Architecture-specific notebooks
├── wacdm/, wcdm/        # Dark energy model experiments
├── input*/              # Data directories
└── models/              # Trained model checkpoints
```

### Running Notebooks
- Use Jupyter Notebook or JupyterLab:
  ```bash
  jupyter notebook
  ```
- Execute notebooks in the appropriate subdirectories (CNN/, Dense/, RNN/, etc.).
- For automated execution, use `papermill`:
  ```bash
  papermill input_notebook.ipynb output_notebook.ipynb -p parameter_name value
  ```

### Running Python Scripts
- Execute scripts directly:
  ```bash
  python Geradores/gerador_pontos.py
  ```
- Use the cosmology package for data generation:
  ```python
  from cosmology.generators import generate_lcdm_data
  generate_lcdm_data(n_simulations=10000, output_dir='input/')
  ```

### Single Test Execution
- No formal test suite currently exists. If tests are added later, use `pytest`:
  ```bash
  pytest path/to/test_file.py::test_function_name -v
  ```
- For automated execution, use `papermill`:
  ```bash
  papermill input_notebook.ipynb output_notebook.ipynb -p parameter_name value
  ```

### Linting & Formatting
- **Recommended tools** (not yet configured):
  - `black` for code formatting
  - `isort` for import sorting
  - `flake8` for style checking
  - `mypy` for type checking (if type hints are added)
- Run manually if installed:
  ```bash
  black --check .
  isort --check .
  flake8 .
  mypy .
  ```

## Code Style Guidelines

### General Principles
- Follow PEP 8.
- Use 4 spaces for indentation (no tabs).
- Keep lines ≤ 80 characters.
- Use meaningful variable names (e.g., `cosmological_parameters` not `cp`).
- Prefer descriptive function names (e.g., `calculate_friedmann_equation`).

### Imports
- Group imports in order: standard library, third-party, local.
- Separate groups with a blank line.
- Use absolute imports (e.g., `from Geradores.gerador_pontos import foo`).
- Avoid wildcard imports (`from module import *`).

### Types & Type Hints
- Add type hints for function signatures when possible.
- Use `typing` module for complex types (e.g., `List[Tuple[float, float]]`).
- For numpy arrays, use `np.ndarray` type hint.

### Docstrings
- Use docstrings for all public functions and classes.
- Follow Google or NumPy docstring style.
- Include parameter descriptions, return values, and examples where helpful.

### Error Handling
- Use try-except blocks for expected errors (e.g., file I/O, network calls).
- Raise custom exceptions for domain-specific errors.
- Log errors appropriately (consider using `logging` module).

### Naming Conventions
- Variables: `snake_case` (e.g., `hubble_constant`).
- Functions: `snake_case` (e.g., `friedmann_equation`).
- Classes: `PascalCase` (e.g., `CosmologyModel`).
- Constants: `UPPER_SNAKE_CASE` (e.g., `OMEGA_M`).

### Notebook Conventions
- Keep notebooks clean: avoid excessive output stored.
- Use markdown cells to explain steps.
- Use consistent cell ordering (top-down execution).
- Save notebooks with cleared output before committing.

### Data Files
- Store raw data in `input/`, `input2/`, `input31/`, `input47/`.
- Use descriptive filenames (e.g., `simulation_batch_1.npy`).
- Do not commit large binary files; use Git LFS if needed.

### Model Files
- Saved models go in `models/` directory.
- Use descriptive subdirectories (e.g., `models/cnn_v1`).
- Include a README in each model directory describing architecture and training.

## Project Structure Overview
- `CNN/`, `Dense/`, `RNN/`: Architecture‑specific notebooks.
- `SVM/`: Support Vector Machine scripts.
- `Geradores/`: Data generation scripts.
- `wacdm/`, `wcdm/`: Variant studies.
- `models/`: Trained model checkpoints.
- `input*/`: Data directories.
- `imagens/`: Generated plots and visualizations.
- `References/`: Supporting documents.

## Additional Notes
- This is a research repository; reproducibility is key.
- When modifying code, ensure that notebook outputs remain consistent.
- If adding new dependencies, update a `requirements.txt` file.
- Prefer functional, modular code over monolithic scripts.
- Use version control for all changes (commit often, push to remote).
- Always test changes in a separate branch before merging to main.

## Contact
- Repository owner: Guilherme de Souza Ramos Cardoso (guilhermesouza1302@gmail.com)
- Supervisor: Luciano Casarini