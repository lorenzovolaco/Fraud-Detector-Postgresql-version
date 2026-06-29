# Detecção de Fraude com Graph Neural Networks (GNN) e PostgreSQL

Este projeto demonstra como usar Inteligência Artificial baseada em Grafos (GNN - Graph Neural Networks) para identificar contas suspeitas e padrões de fraude em transações financeiras.

O sistema conecta diretamente a um banco de dados PostgreSQL, extrai os dados via SQL, constrói um grafo de transações e treina um modelo de GCN para detectar fraudes automaticamente.

## 📌 O que o projeto faz?

Conecta ao banco PostgreSQL
Extrai dados de contas e transações usando SQL.
Modela os dados como um grafo
Nós → Contas bancárias
Arestas → Transações financeiras
Feature Engineering (PageRank e métricas estruturais)
Calcula métricas da rede para enriquecer os dados.
Treina uma GNN (GCN)
O modelo aprende padrões de comportamento e classifica cada conta como:
Normal
Fraude
Avalia o modelo
Utiliza F1-score, ideal para dados desbalanceados.
Visualiza o grafo com predições
Mostra a rede de transações com as classificações do modelo.
## 🛠️ Instalação

Para rodar o projeto, instale as dependências:

```bash
pip install torch torch-geometric networkx numpy matplotlib scikit-learn psycopg
