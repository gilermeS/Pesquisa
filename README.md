# Análise de Dados Cosmológicos com Deep Learning — Projeto Planck

**Autor**: Guilherme de Souza Ramos Cardoso  
**Orientador**: Luciano Casarini  
**Instituição**: Universidade Federal de Sergipe (UFS)  
**Departamento**: Departamento de Física  
**Ano**: 2025

## Descrição

Repositório com implementação e análise de modelos de Machine Learning e Deep Learning aplicados a dados cosmológicos (dados sintéticos gerados por Monte Carlo Bootstrap). O objetivo é comparar arquiteturas e avaliar a capacidade de reconstrução de parâmetros cosmológicos (ex.: H0, omega_m).

## Estrutura do projeto

 - input/         : Dados de entrada (simulações)
 - input2/        : Dados adicionais
 - input31/        : Dados adicionais com 31 pontos e dados reais retirados da Tabela 1 de Bengaly et al.
 - imagens/       : Gráficos e visuais gerados
 - models/        : Modelos treinados (saved_model)
 - CNN/, Dense/, RNN/: Notebooks e experimentos por arquitetura
 - wacdm/, wcdm/  : Variantes específicas do estudo
 - pesquisa2parte/ : Análises complementares e modelos TPOT

## Modelos incluídos

- Deep Learning:
   - Dense (fully connected)
   - CNN (convolutional)
   - RNN (GRU / bidirecional)

- Machine Learning clássico:
   - SVM
   - Análise de importância por permutação

## Requisitos (rápido)

 - Python 3.8+  
 - TensorFlow (>=2.10)  
 - scikit-learn  
 - numpy, pandas, matplotlib, tqdm, joblib

Recomendo criar um ambiente virtual e instalar dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install tensorflow scikit-learn numpy pandas matplotlib tqdm joblib
```

## Como usar (rápido)

- Abra os notebooks correspondentes (`CNN/CNN.ipynb`, `Dense/Dense.ipynb`, `RNN/RNN.ipynb`) e execute as células para treinar ou avaliar modelos.  
- Para carregar modelos já treinados, veja a pasta `models/` — por exemplo `models/rnn`, `models/cnn`, `models/dense`.  
- O notebook `feature_importance.ipynb` mostra como calcular importância por permutação (para as redes) e também usa um `SVM` salvo (`models/svm.pkl`).

## Executando um experimento (exemplo)

1. Gerar/colocar os arquivos `.npy` na pasta `input/` (nome padrão: `data_1.npy`, `data_2.npy`, ...).  
2. Abrir o notebook desejado e rodar as células (ou usar `papermill` para execução programática).  

## Dados

Os dados de entrada são arrays NumPy em `input/`, `input2/`e `input31/`. Cada arquivo `data_*.npy` contém os vetores usados como features e alvo.

## Modelos salvos

Modelos treinados são exportados em `models/` no formato Keras `saved_model`. Exemplos:

- `models/rnn`  
- `models/rnn_bi`  
- `models/cnn`  
- `models/dense`

## Sugestões / próximos passos

- Gerar um `requirements.txt` para fixar versões (posso criar se desejar).  
- Automatizar execução de notebooks com `papermill` para reprodutibilidade.

## Licença

MIT

## Contato

guilhermesouza1302@gmail.com
