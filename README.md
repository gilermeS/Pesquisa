# Análise de Dados Cosmológicos com Deep Learning: Aplicação ao Projeto Planck

**Autor**: Guilherme de Souza Ramos Cardoso  
**Orientador**: Luciano Casarini  
**Instituição**: Universidade Federal de Sergipe (UFS)  
**Departamento**: Departamento de Física  
**Ano**: 2025

## 📌 Descrição do Projeto

Este repositório contém a implementação e análise de diferentes modelos de Machine Learning e Deep Learning aplicados à análise de dados cosmológicos do Projeto Planck. O estudo utiliza dados sintéticos gerados via Monte Carlo Bootstrap para avaliar o desempenho de diversos modelos na reconstrução e predição de parâmetros cosmológicos.

## 🏗️ Estrutura do Projeto

```
.
├── input/                  # Dados de entrada originais
├── input2/                 # Dados de entrada adicionais
├── imagens/               # Visualizações e gráficos gerados
├── models/                # Modelos treinados salvos
├── wacdm/                 # Implementações específicas do modelo wCDM
├── pesquisa2parte/        # Análises complementares
```

## 🧠 Modelos Implementados

O projeto implementa e compara diversos modelos de aprendizado de máquina:

1. **Deep Learning**
   - Redes Neurais Densas (Dense.ipynb, Dense_Kfold.ipynb)
   - Redes Neurais Convolucionais (CNN.ipynb, CNN_Kfold.ipynb)
   - Redes Neurais Recorrentes (RNN.ipynb, RNN_Kfold.ipynb)

2. **Machine Learning Tradicional**
   - Support Vector Machines (SVM.ipynb)
   - Análise de Importância de Features (feature_importance.ipynb)

## 📊 Análises e Ferramentas

- **Validação Cruzada**: Implementação de K-Fold Cross Validation para todos os modelos
- **Análise de Features**: Avaliação da importância das features usando diferentes metodologias
- **Geração de Dados**: Scripts para geração de dados sintéticos (gerador_pontos.py, gerador_pontos.ipynb)

## 🔍 Principais Resultados

- Comparação abrangente entre modelos de Deep Learning (ANNs, CNNs, RNNs) e métodos tradicionais
- Análise detalhada do impacto de hiperparâmetros no desempenho dos modelos
- Avaliação da importância de features para diferentes abordagens
- Demonstração do potencial do Deep Learning em complementar métodos tradicionais em cosmologia

## 🛠️ Tecnologias Utilizadas

- Python
- TensorFlow/Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebooks

## 📜 Licença

Este projeto está licenciado sob a licença MIT.

## 📞 Contato

Para dúvidas ou colaborações, entre em contato:
- Email: guilhermesouza1302@gmail.com
