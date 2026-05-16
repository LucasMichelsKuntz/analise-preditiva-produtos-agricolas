# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
  <a href="https://www.fiap.com.br/">
    <img src="assets/logo-fiap.png" alt="FIAP" border="0" width="40%" height="40%">
  </a>
</p>

<br>

# Análise Exploratória e Modelos Preditivos — Produtos Agrícolas

## Grupo 71

## Integrantes

- <a href="https://www.linkedin.com/in/">Lucas Michels Kuntz — RM 570050</a>
- <a href="https://www.linkedin.com/in/">João Pedro Alencar — RM 573473</a>
- <a href="https://www.linkedin.com/in/">Alisson Vinicius de Souza Rabelo Teixeira — RM 573512</a>

## Professores

### Tutor(a)
- <a href="https://www.linkedin.com/in/">Sabrina Otoni</a>

### Coordenador(a)
- <a href="https://www.linkedin.com/in/">André Godoi</a>

---

## Descrição

Este repositório contém a entrega da atividade da disciplina **Fase 3 — Cap 10: A primeira técnica de aprendizado de máquina** (FIAP — Turma 1TIAOA). O trabalho aplica um pipeline completo de ciência de dados — desde a exploração inicial dos dados até a comparação de cinco algoritmos de Machine Learning — sobre um dataset real de condições agronômicas.

### Problema e Objetivo

Dado um conjunto de medições de solo e clima de uma determinada região, **qual cultura agrícola tem maior probabilidade de prosperar naquelas condições?** Esse é um problema de classificação multiclasse: o modelo recebe sete variáveis numéricas e deve prever uma entre 22 culturas possíveis.

O dataset `produtos_agricolas.csv` reúne **2.200 registros**, distribuídos igualmente entre as 22 culturas (100 amostras cada), com as seguintes variáveis:

| Variável | Descrição |
|---|---|
| `N` | Teor de nitrogênio no solo (mg/kg) |
| `P` | Teor de fósforo no solo (mg/kg) |
| `K` | Teor de potássio no solo (mg/kg) |
| `temperature` | Temperatura média da região (°C) |
| `humidity` | Umidade relativa do ar (%) |
| `ph` | pH do solo |
| `rainfall` | Precipitação média (mm) |
| `label` | Cultura agrícola (variável alvo) |

### O que foi desenvolvido

**1. Análise Exploratória (EDA)**

Verificação da integridade dos dados (zero valores ausentes, zero duplicatas), tipos de variáveis, estatísticas descritivas por feature e distribuição do target. A EDA revelou que o dataset é perfeitamente balanceado — condição favorável ao treinamento de classificadores sem necessidade de técnicas de resampling.

**2. Análise Descritiva com 9 gráficos**

- Distribuição das 22 culturas (confirmação do balanceamento)
- Mapa de correlação entre as 7 features (maior correlação: K×P ≈ 0,74; demais abaixo de 0,5)
- Box plots de todas as features por cultura — identificação de culturas com perfis extremos (ex.: `rice` com alta precipitação; `apple` com baixa temperatura)
- Histogramas com KDE para cada feature — distribuições bimodais em temperatura e umidade indicam dois grupos climáticos bem distintos (temperado × tropical)
- Scatter temperatura × precipitação por cultura — separação visual clara entre quatro clusters climáticos
- Radar chart com perfil normalizado de três culturas selecionadas versus o perfil ideal

**3. Perfil Ideal de Solo/Clima e Comparação de Culturas**

O **perfil ideal** foi definido como a mediana global de todas as variáveis (N≈37, P≈51, K≈32, temp≈25,6 °C, umidade≈80,5%, pH≈6,43, precipitação≈94,9 mm) e representa o equilíbrio médio entre as exigências das 22 culturas. Três culturas foram escolhidas por contrastarem significativamente entre si:

- **Arroz (Rice):** exige o dobro de nitrogênio (+116%) e 2,5× mais chuva (+146%) que o perfil ideal. Perfil de região tropical úmida de alta demanda hídrica.
- **Café (Coffee):** surpreende com alto consumo de nitrogênio (+178%), mas rejeita umidade excessiva (−28%). Não é uma cultura moderada — é exigente em N com ambiente mais seco.
- **Maçã (Apple):** exige solo extremamente mineralizado — potássio +525% e fósforo +168% acima do ideal — com temperatura mais fria (−12%) e pH mais ácido (−8%). O diferencial real está na composição do solo, não apenas no clima.

**4. Cinco Modelos Preditivos**

Todos os modelos seguiram boas práticas de ML: split treino/teste estratificado (80/20), `StandardScaler` ajustado exclusivamente no treino (sem data leakage) e validação cruzada estratificada de 5 folds para estimativa robusta de generalização.

| # | Algoritmo | Acurácia Teste | F1-Macro | Acurácia CV (5-fold) |
|---|---|---|---|---|
| 1 | Regressão Logística | 97,27% | 97,25% | 96,82% |
| 2 | Árvore de Decisão | 97,95% | 97,94% | 98,52% |
| 3 | **Random Forest** | **99,55%** | **99,55%** | **99,43%** |
| 4 | KNN (k=5) | 97,95% | 97,93% | 96,53% |
| 5 | Gradient Boosting | 98,86% | 98,86% | 98,58% |

O **Random Forest** (200 árvores) obteve o melhor desempenho geral com 99,55% de acurácia e gap de apenas 0,12 pp entre teste e CV — sem indício de overfitting. Notável: a Regressão Logística, o modelo mais simples, ainda atingiu 97,27%, evidenciando forte separabilidade linear entre as culturas. A análise de importância de features aponta `rainfall` e `humidity` como as mais discriminativas, seguidas de `K` e `temperature`.

---

## Estrutura de Pastas

```
fiap-fase3-cap10/
├── .github/               # Configurações do GitHub (workflows, templates)
├── assets/                # Imagens e arquivos não-estruturados
│   └── logo-fiap.png
├── config/                # Arquivos de configuração do projeto
├── document/              # Documentação e dados
│   ├── Atividade_Cap10_produtos_agricolas.csv
│   └── other/             # Documentos complementares
├── scripts/               # Scripts auxiliares
├── src/                   # Código-fonte principal
│   └── LucasMichelsKuntz_RM570050_fase3_cap10.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Como Executar o Notebook

### Pré-requisitos

- Python 3.9+
- Jupyter Notebook ou JupyterLab (ou VS Code com extensão Jupyter)

### Instalação

```bash
# Clone o repositório
git clone https://github.com/<seu-usuario>/fiap-fase3-cap10.git
cd fiap-fase3-cap10

# Crie e ative um ambiente virtual (recomendado)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### Executando

```bash
# Recomendado: execute a partir da raiz do repositório
jupyter lab src/LucasMichelsKuntz_RM570050_fase3_cap10.ipynb
```

O notebook localiza automaticamente o CSV em `document/` — sem nenhuma dependência de caminhos externos à pasta do repositório.

### Execução Alternativa — Google Colab

1. Faça upload dos arquivos `.ipynb` e `document/Atividade_Cap10_produtos_agricolas.csv`.
2. Adicione uma célula no início com `!mkdir -p document && mv Atividade_Cap10_produtos_agricolas.csv document/`.
3. Execute todas as células em ordem (`Runtime > Run all`).

---

## Resultados Principais

Obtidos com `SEED=42`, split 80/20 estratificado, `StratifiedKFold(n_splits=5)`:

| Modelo | Acurácia Teste | F1-Macro | CV 5-fold |
|---|---|---|---|
| **Random Forest** | **99,55%** | **99,55%** | **99,43%** |
| Gradient Boosting | 98,86% | 98,86% | 98,58% |
| Árvore de Decisão | 97,95% | 97,94% | 98,52% |
| KNN (k=5) | 97,95% | 97,93% | 96,53% |
| Regressão Logística | 97,27% | 97,25% | 96,82% |

**Features mais discriminativas (Random Forest):** `rainfall` > `humidity` > `K` > `temperature` > `N` > `P` > `ph`

---

## Tecnologias Utilizadas

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4C72B0)

---

## Histórico de Lançamentos

- **0.1.0** — 16/05/2026 — Versão inicial: EDA, análise descritiva e 5 modelos de ML

---

## Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1">

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1) — FIAP
