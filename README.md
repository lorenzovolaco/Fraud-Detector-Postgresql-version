Fraud Detection with Graph Neural Networks (GNN) + PostgreSQL

Projeto de detecção de fraudes em redes financeiras utilizando Graph Neural Networks (GCN), com dados extraídos diretamente de um banco PostgreSQL.

📌 Visão geral

Este sistema modela transações financeiras como um grafo, onde:

Nós representam contas
Arestas representam transações
Um modelo GNN aprende padrões de comportamento suspeito
⚙️ Pipeline
Extração de dados (SQL / PostgreSQL)
Contas e transações são carregadas diretamente do banco.
Construção do grafo
Conversão dos dados para estrutura compatível com PyTorch Geometric.
Feature engineering
Adição de métricas estruturais (ex: PageRank).
Treinamento do modelo
GCN aprende a classificar nós como:
Normal (0)
Fraude (1)
Visualização
Grafo final com previsões do modelo.
🧠 Tecnologias
Python
PyTorch
PyTorch Geometric
PostgreSQL (psycopg)
NetworkX
NumPy
Matplotlib
Scikit-learn
📦 Instalação
pip install torch torch-geometric networkx numpy matplotlib scikit-learn psycopg
🚀 Execução

Configure as credenciais do PostgreSQL no código e execute:

python main.py
📊 Saída do modelo
🔵 Classe 0 → Conta normal
🔴 Classe 1 → Fraude
Métrica de avaliação: F1-score
Visualização do grafo com predições
