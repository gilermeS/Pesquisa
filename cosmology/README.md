# Cosmology Package

Pacote Python para modelagem cosmológica e geração de dados sintéticos.

## Estrutura

```
cosmology/
├── __init__.py       # Inicialização do pacote
├── parameters.py     # Parâmetros cosmológicos (Planck 2018)
├── models.py         # Modelos cosmológicos (Friedmann equations)
└── generators.py     # Gerador de dados Monte Carlo
```

## Modelos Disponíveis

### 1. LCDM (Lambda-CDM)
Modelo padrão da cosmologia com constante cosmológica.

```python
from cosmology import CosmologicalModel

model = CosmologicalModel('LCDM')
H_z = model.hubble_parameter(z)  # z pode ser escalar ou array
```

### 2. wCDM
Modelo com equação de estado constante para energia escura.

```python
model = CosmologicalModel('wCDM', parameters=CosmologicalParameters.wcdm(w0=-1.1))
```

### 3. w(a)CDM (Parâmetrização CPL)
Modelo com evolução da equação de estado.

```python
model = CosmologicalModel('wACDM', parameters=CosmologicalParameters.wacdm(w0=-0.9, wa=-0.5))
```

## Geração de Dados

### Uso Básico

```python
from cosmology.generators import generate_lcdm_data

# Gerar 10,000 simulações LCDM
generator = generate_lcdm_data(
    n_simulations=10000,
    n_redshift_points=80,
    output_dir='input/'
)
```

### Usando a Classe MonteCarloGenerator

```python
from cosmology.generators import MonteCarloGenerator
from cosmology.parameters import CosmologicalParameters

# Parâmetros personalizados
params = CosmologicalParameters(
    omega_m=0.315,
    sigma_omega_m=0.007,
    h0=67.45,
    sigma_h0=0.62
)

generator = MonteCarloGenerator(
    model_type='LCDM',
    parameters=params,
    n_simulations=5000,
    n_redshift_points=31,
    output_dir='custom_input/'
)

generator.generate_all_simulations()
```

## Formato dos Dados

Os dados são salvos como arrays NumPy com a seguinte estrutura:

### LCDM (4 colunas)
```
[z_0, H(z_0), n*H0, n*Ω_m]
[z_1, H(z_1), n*H0, n*Ω_m]
...
[z_n, H(z_n), n*H0, n*Ω_m]
```

### wCDM (5 colunas)
```
[z_0, H(z_0), n*H0, n*Ω_m, n*w0]
...
```

### wACDM (6 colunas)
```
[z_0, H(z_0), n*H0, n*Ω_m, n*w0, n*wa]
...
```

Onde:
- `n` é o número de pontos de redshift
- Os valores de parâmetros são replicados `n` vezes para facilitar o carregamento

## Equações Implementadas

### LCDM
```
H(z) = H0 * sqrt(Ω_m * (1+z)³ + (1-Ω_m))
```

### wCDM
```
H(z) = H0 * sqrt(Ω_m * (1+z)³ + (1-Ω_m) * (1+z)^(3(1+w0)))
```

### wACDM (CPL)
```
H(z) = H0 * sqrt(Ω_m * (1+z)³ + (1-Ω_m) * (1+z)^(3(1+w0+wa)) * exp(-3*wa*z/(1+z)))
```

## Dependências

- numpy >= 1.21.0
- tqdm (opcional, para barras de progresso)

## Exemplo Completo

```python
import numpy as np
from cosmology.generators import generate_lcdm_data, generate_wcdm_data
from cosmology.models import CosmologicalModel
from cosmology.parameters import CosmologicalParameters

# Gerar dados LCDM
lcdm_gen = generate_lcdm_data(n_simulations=1000, output_dir='lcdm_data/')

# Gerar dados wCDM
wcdm_params = CosmologicalParameters.wcdm(w0=-1.2)
wcdm_gen = generate_wcdm_data(
    w0=-1.2,
    n_simulations=1000,
    output_dir='wcdm_data/'
)

# Calcular H(z) para uma cosmologia específica
model = CosmologicalModel('LCDM')
z = np.linspace(0.1, 1.5, 80)
H_z = model.hubble_parameter(z)

print(f"H(z=0.5) = {H_z[20]:.2f} km/s/Mpc")
```

## Testes

Para testar o pacote:

```bash
python -m pytest cosmology/tests/ -v
```

## Autor

Guilherme de Souza Ramos Cardoso  
Orientador: Luciano Casarini  
Universidade Federal de Sergipe (UFS)