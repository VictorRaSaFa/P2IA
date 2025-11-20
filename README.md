# 🍌 Classificação da Qualidade de Bananas – IAP2  
Repositório oficial do projeto de classificação de qualidade de bananas utilizando Machine Learning.  
Este projeto faz parte da disciplina de Inteligência Artificial.

---

## 📦 Sobre o Projeto

O objetivo é prever a qualidade de bananas classificando-as como:

- **Good**
- **Bad**

O modelo utiliza um pipeline completo com:

- carregamento e inspeção de dados  
- pré-processamento (limpeza, encoding, remoção de outliers)
- normalização  
- divisão em treino / validação / teste  
- treinamento com Random Forest  
- avaliação (acurácia e matriz de confusão)  
- predição de amostra real  

Toda a implementação está no notebook:
IAP2/
├── banana_classifier.ipynb # Notebook principal
├── banana_quality.csv # Dataset
├── .gitignore # Arquivos ignorados no Git
├── tp.py # Código completo
├── requirements.txt # Dependências
└── README.md # Documentação do repositório

---

## 🚀 Como executar

### 1. Clone o repositório


`git clone https://github.com/VictorRaSaFa/IAP2`

`cd IAP2`

### 2. Crie um ambiente virtual (Opcional)

`python -m venv venv`

### 3. Instale as dependências

`pip install -r requirements.txt`

### 4. Abra o Jupyter

`python -m notebook`

e selecione

`banana_classifier.ipynb`

## 📊 Resultados Gerados pelo Notebook

Estatísticas descritivas

Distribuição da classe Quality

Matriz de confusão com heatmap

Acurácia da validação

Predição de uma amostra real

## 🧠 Modelo Utilizado

RandomForestClassifier

## 📑 Dataset

O dataset banana_quality.csv contém atributos numéricos representando características das bananas que culminam na coluna Quality com rótulos Good ou Bad.

## 🛠 Tecnologias

Python 3.x

Pandas

NumPy

Matplotlib

Seaborn

Scikit-Learn

Jupyter Notebook
