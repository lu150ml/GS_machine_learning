# GS Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/XGBoost-EC4D37?logo=xgboost&logoColor=white" alt="XGBoost" />
  <img src="https://img.shields.io/badge/LightGBM-013243?logoColor=white" alt="LightGBM" />
  <img src="https://img.shields.io/badge/Seaborn-4E79A7?logoColor=white" alt="Seaborn" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?logoColor=white" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/SHAP-FF6F61?logoColor=white" alt="SHAP" />
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white" alt="Google Colab" />
</p>

Projeto acadêmico focado em treinar modelos de classificação para prever se startups serão **adquiridas ou fecharão**. Todo o fluxo — da análise exploratória ao pós-modelagem — está disponível no notebook [`gs_machine_learning.ipynb`](gs_machine_learning.ipynb), que pode ser executado tanto localmente quanto no Google Colab.

## 📊 Visão geral
- **Fonte dos dados**: arquivo `startup data (1).csv`, contendo informações históricas e financeiras de startups (financiamentos, marcos, localização, etc.).
- **Objetivo**: estimar a probabilidade de aquisição a partir de atributos numéricos e categóricos.
- **Output principal**: modelos treinados e interpretados, com destaque para um **XGBoost otimizado** e um **ensemble soft voting** que combinam XGBoost, SVM e MLP.

## ✨ Principais etapas do notebook
1. **Importação e checagens iniciais** – leitura do CSV, inspeção de forma, amostras e colunas.
2. **Análise exploratória** – contagens normalizadas de `status`, distribuições por estado/categoria e auditoria de valores ausentes.
3. **Limpeza e pré-processamento** – remoção de colunas redundantes, imputação de campos temporais e normalização pontual.
4. **Feature engineering avançada** – criação de métricas como `funding_per_round`, `funding_span`, `num_rounds_active`, `rel_invest_ratio` e `vc_investment_strength`.
5. **Treinamento de modelos** – Decision Tree como baseline e modelos avançados (XGBoost, SVM, MLP e ensemble) com busca de hiperparâmetros via `GridSearchCV`.
6. **Avaliação** – métricas de acurácia, classification report e matriz de confusão, além de comparativos visuais.
7. **Interpretabilidade** – importância de features com XGBoost e explicações globais com SHAP, incluindo estudos de caso (índices 331–552).

## 📁 Estrutura
```
├── README.md                     # Este guia
└── gs_machine_learning.ipynb     # Notebook principal com todo o pipeline
```

## 🧱 Pré-requisitos
- Python 3.10+
- Pip ou outro gerenciador de pacotes compatível

Dependências principais (instaladas no notebook ou manualmente):
```
pandas numpy scikit-learn xgboost lightgbm seaborn matplotlib shap scipy
```

## 🚀 Passo a passo para executar localmente
1. **Clone o repositório**
   ```bash
   git clone https://github.com/lu150ml/GS_machine_learning.git
   cd GS_machine_learning
   ```
2. **Crie um ambiente virtual (opcional, mas recomendado)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate    # Windows
   ```
3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt  # se você criar um arquivo
   # ou instale diretamente:
   pip install pandas numpy scikit-learn xgboost lightgbm seaborn matplotlib shap scipy
   ```
4. **Disponibilize o dataset**
   - Baixe `startup data (1).csv` e coloque-o no diretório raiz ou ajuste o caminho na constante `PATH` logo no início do notebook.
5. **Execute o notebook**
   ```bash
   jupyter notebook gs_machine_learning.ipynb
   ```
   Em seguida, rode as células na ordem para reproduzir todo o fluxo.

## ☁️ Executando no Google Colab
1. Clique no badge "Open in Colab" presente na primeira célula do notebook (ou [acesse diretamente](https://colab.research.google.com/github/lu150ml/GS_machine_learning/blob/main/gs_machine_learning.ipynb)).
2. Faça upload do arquivo `startup data (1).csv` para o ambiente Colab (ou monte seu Google Drive e ajuste o `PATH`).
3. Execute as células sequencialmente:
   - Instalação de dependências extras (caso solicitado).
   - Importação de bibliotecas e leitura dos dados.
   - EDA, engenharia de atributos e treinamento dos modelos.
4. Utilize os gráficos gerados (matriz de confusão, comparação de acurácia, plots SHAP) para interpretar os resultados.

## 📈 Resultados e insights
- **Modelos avançados**: o XGBoost otimizado apresentou o melhor equilíbrio entre precisão e interpretabilidade, sendo usado como base para explicações SHAP.
- **Ensemble soft voting**: ao ponderar XGBoost (peso 3) com SVM e MLP (peso 1 cada), obteve-se uma melhora marginal na acurácia e maior robustez.
- **Principais drivers**: variáveis temporais (`age_last_milestone_year`, `age_first_funding_year`), intensidade de investimento (`funding_total_usd`, `funding_rounds`) e relacionamentos (`relationships`, `avg_participants`) foram determinantes para classificar casos reais do dataset.

## 🛠️ Próximos passos sugeridos
- Criar um arquivo `requirements.txt` formal para simplificar a instalação.
- Salvar modelos treinados (pickle) e expor uma API leve para scoring.
- Integrar novas fontes de dados (ex.: métricas de tração atualizadas) e testar técnicas de balanceamento como SMOTE.

Sinta-se à vontade para abrir *issues* ou enviar *pull requests* com melhorias! 💡
