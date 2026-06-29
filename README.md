Fraud Detection with Graph Neural Networks (GNN) + PostgreSQL

Este projeto realiza detecção de fraudes em redes financeiras utilizando Graph Neural Networks (GCN) 🧠🔗, modelando transações bancárias como um grafo para capturar relações complexas entre contas.

📊 Como funciona o sistema:

🗄️ Dados são extraídos diretamente de um banco PostgreSQL
🔵 Nós representam contas bancárias
🔗 Arestas representam transações financeiras
📈 Features estruturais (como PageRank) são adicionadas ao grafo
🧠 Um modelo GCN é treinado para identificar padrões de fraude

⚙️ Pipeline do projeto:

📥 Extração de dados via SQL (PostgreSQL)
🕸️ Construção do grafo com NetworkX / PyTorch Geometric
🧪 Feature engineering (PageRank e métricas estruturais)
🤖 Treinamento do modelo GNN (classificação binária)
📊 Avaliação com F1-score
🎨 Visualização do grafo com predições

🎯 Classes do modelo:

🔵 Classe 0 → Conta normal
🔴 Classe 1 → Fraude

🛠️ Tecnologias utilizadas:
Python 🐍 | PyTorch 🔥 | PyTorch Geometric 🧠 | PostgreSQL 🗄️ | psycopg | NetworkX 🔗 | NumPy ➗ | Matplotlib 📊 | Scikit-learn 🤖

📈 Resultado final:
O sistema gera uma visualização do grafo com as predições do modelo, facilitando a identificação de comportamentos suspeitos na rede financeira.
