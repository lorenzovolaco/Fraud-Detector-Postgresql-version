import psycopg
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def load_financial_data_from_postgres():
    print("-> Conectando ao PostgreSQL e buscando dados...")
    
    # abre a conexao com o banco usando o driver psycopg
    conn = psycopg.connect(
        host="nomehost",          
        dbname="nomedb",      
        user="usuario",        
        password="senha",      
        port="5432",                
    )
    cursor = conn.cursor()

    # roda a query para puxar os nos (contas), suas features e a classe alvo
    cursor.execute("SELECT account_id, feat_1, feat_2, feat_3, fraude FROM accounts;")
    nodes_raw = cursor.fetchall()
    
    # mapeia os ids reais do postgres para indices de 0 a n-1 exigidos pela gnn
    node_id_map = {row[0]: idx for idx, row in enumerate(nodes_raw)}
    
    x_list = []
    y_list = []
    for row in nodes_raw:
        x_list.append([row[1], row[2], row[3]])  # separa os atributos numéricos em x
        y_list.append(row[4])                    # separa o target (0 ou 1) em y
        
    # transforma os dados coletados do banco em tensores do pytorch
    x = torch.tensor(x_list, dtype=torch.float)
    y = torch.tensor(y_list, dtype=torch.long)

    # roda a query para buscar as arestas(transações)
    cursor.execute("SELECT source_id, target_id FROM transactions;")
    edges_raw = cursor.fetchall()
    
    edge_list = []
    for src, tgt in edges_raw:
        # mapeia as conexoes usando os novos indices sequenciais do dicionario
        if src in node_id_map and tgt in node_id_map:
            edge_list.append([node_id_map[src], node_id_map[tgt]])
            
    # converte para tensor e ajusta o shape para [2, num_edges] conforme o padrao do pyg
    edges = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # encerra o cursor e a sessao com o postgres de forma limpa
    cursor.close()
    conn.close()
    
    return Data(x=x, edge_index=edges, y=y)


def extract_graph_features(data):
    print("-> Calculando PageRank das transações...")
    G = nx.DiGraph()
    G.add_edges_from(data.edge_index.t().numpy())

    pagerank = nx.pagerank(G)
    pr_values = [pagerank.get(i, 0.0) for i in range(data.num_nodes)]

    pr_tensor = torch.tensor(pr_values, dtype=torch.float).view(-1, 1)
    data.x = torch.cat([data.x, pr_tensor], dim=1)
    return data


class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        embeddings = x 
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), embeddings


def train_model(data):
    print("-> Iniciando treinamento da GNN...")
    learning_rate = 0.01
    epochs = 51
    hidden_channels = 16

    model = FraudGNN(
        in_channels=data.num_node_features,
        hidden_channels=hidden_channels,
        out_channels=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

        preds = out.argmax(dim=1)
        f1 = f1_score(data.y.numpy(), preds.numpy(), zero_division=0)

        if epoch % 10 == 0:
            print(f"Época {epoch:02d} | Custo (Loss): {loss:.4f} | Métrica F1-Score: {f1:.4f}")

    return model


def visualize_fraud_ring(data, model):
    print("-> Renderizando o grafo final...")
    model.eval()

    with torch.no_grad():
        out, _ = model(data.x, data.edge_index)
        preds = out.argmax(dim=1).numpy()

    G = nx.DiGraph()
    G.add_edges_from(data.edge_index.t().numpy())

    color_map = ["red" if preds[i] == 1 else "blue" for i in G.nodes()]

    plt.figure(figsize=(10, 8))
    plt.title("Detecção de Fraude em Rede (Dados do PostgreSQL)")

    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos,
        node_color=color_map,
        with_labels=True,
        node_size=800,
        edge_color="gray"
    )

    import matplotlib.patches as mpatches
    plt.legend(handles=[
        mpatches.Patch(color='red', label='Previsão: Fraude'),
        mpatches.Patch(color='blue', label='Previsão: Normal')
    ])
    plt.show()


if __name__ == "__main__":
    graph_data = load_financial_data_from_postgres()
    graph_data = extract_graph_features(graph_data)
    model = train_model(graph_data)
    visualize_fraud_ring(graph_data, model)
