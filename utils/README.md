# Utils Package

Pacote de funções utilitárias para análise de dados cosmológicos.

## Estrutura

```
utils/
├── __init__.py       # Inicialização do pacote
├── data_loading.py   # Carregamento otimizado de dados
├── visualization.py  # Funções de visualização
└── model_utils.py    # Utilidades para modelos
```

## Funcionalidades

### 1. Carregamento de Dados (`data_loading.py`)

#### Carregamento Sequencial
```python
from utils.data_loading import load_dataset

X, y = load_dataset('input/', n_files=1000)
```

#### Carregamento Paralelo
```python
from utils.data_loading import load_dataset_parallel

X, y = load_dataset_parallel('input/', n_jobs=-1)  # Usa todos os cores
```

#### Carregamento com Cache
```python
from utils.data_loading import load_dataset_cached

# Primeira chamada carrega do disco
# Chamadas subsequentes usam cache
X, y = load_dataset_cached('input/', cache_dir='.cache')
```

#### Normalização
```python
from utils.data_loading import normalize_dataset

X_norm, y_norm, params = normalize_dataset(X, y, method='standard')
```

### 2. Visualização (`visualization.py`)

#### Histórico de Treinamento
```python
from utils.visualization import plot_training_history

fig = plot_training_history(history.history, save_path='training_history.png')
```

#### Predições
```python
from utils.visualization import plot_predictions

fig = plot_predictions(y_true, y_pred, save_path='predictions.png')
```

#### Dados Cosmológicos
```python
from utils.visualization import plot_cosmological_data

fig = plot_cosmological_data(X, y, n_samples=5)
```

### 3. Utilidades de Modelos (`model_utils.py`)

#### Resumo do Modelo
```python
from utils.model_utils import save_model_summary

summary = save_model_summary(model, save_path='model_summary.json')
```

#### Contagem de Parâmetros
```python
from utils.model_utils import count_parameters

params = count_parameters(model)
print(f"Total parameters: {params['total']:,}")
```

#### Comparação de Modelos
```python
from utils.model_utils import compare_models

comparison = compare_models([model1, model2, model3], 
                           model_names=['CNN', 'RNN', 'Dense'])
```

## Exemplos de Uso

### Pipeline Completo de Análise

```python
import numpy as np
from utils.data_loading import load_dataset_cached, normalize_dataset
from utils.visualization import plot_cosmological_data, plot_predictions
from utils.model_utils import save_model_summary

# 1. Carregar dados com cache
X, y = load_dataset_cached('input/', cache_dir='.cache')

# 2. Normalizar
X_norm, y_norm, norm_params = normalize_dataset(X, y, method='standard')

# 3. Visualizar
plot_cosmological_data(X, y, n_samples=10, save_path='data_samples.png')

# 4. Treinar modelo (exemplo)
# model = create_model()
# model.fit(X_train, y_train, ...)

# 5. Salvar resumo
save_model_summary(model, save_path='model_info.json')
```

### Análise de Performance

```python
import time
from utils.data_loading import load_dataset, load_dataset_parallel

# Testar performance
start = time.time()
X1, y1 = load_dataset('input/', verbose=False)
seq_time = time.time() - start

start = time.time()
X2, y2 = load_dataset_parallel('input/', n_jobs=-1, verbose=False)
par_time = time.time() - start

print(f"Speedup: {seq_time/par_time:.2f}x")
```

## Otimizações Implementadas

### 1. Cache Inteligente
- Cache baseado em hash do diretório e parâmetros
- Validação automática de cache obsoleto
- Compactação automática para grandes conjuntos de dados

### 2. Carregamento Paralelo
- Processamento em paralelo com `joblib`
- Balanceamento automático de carga
- Fallback para carregamento sequencial se `joblib` não estiver instalado

### 3. Otimizações de Memória
- Carregamento sob demanda quando possível
- Normalização in-place quando apropriado
- Liberação automática de memória

## Formato dos Dados

### Entrada Esperada
Arquivos `.npy` com estrutura:
```
[z_0, H(z_0), n*H0, n*Ω_m, ...]
[z_1, H(z_1), n*H0, n*Ω_m, ...]
...
```

### Saída
- `X`: Array de shape `(n_samples, n_points, n_features)`
- `y`: Array de shape `(n_samples,)` com valores H0

## Configuração

### Variáveis de Ambiente
```python
import os

# Configurar cache
os.environ['COSMOLOGY_CACHE_DIR'] = '/path/to/cache'

# Configurar número de workers paralelos
os.environ['COSMOLOGY_N_JOBS'] = '4'
```

## Dependências

- numpy >= 1.21.0
- matplotlib >= 3.5.0
- joblib (opcional, para carregamento paralelo)
- scikit-learn (para algumas utilidades)

## Testes

```bash
# Testar carregamento de dados
python -c "from utils.data_loading import load_dataset; print('OK')"

# Testar visualização
python -c "from utils.visualization import plot_training_history; print('OK')"
```

## Autor

Guilherme de Souza Ramos Cardoso  
Orientador: Luciano Casarini  
Universidade Federal de Sergipe (UFS)