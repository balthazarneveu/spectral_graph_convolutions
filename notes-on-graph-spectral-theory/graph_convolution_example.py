import networkx as nx
import matplotlib.pyplot as plt
from random import choice, random

import torch
from torch_geometric.nn import ChebConv

def create_simple_graph(nbNodes=5):
    G = nx.Graph()
    #G.add_nodes_from(range(nbNodes))
    #random_position = list(range(nbNodes))

    """for i in range(nbNodes):
        rd_weight = random()
        rd_node = choice(random_position)
        if i == rd_node:
            rd_node = choice(random_position)
        G.add_node(i, feat=[i])
        G.add_edge(rd_node, i, weight=rd_weight) """
    
    G.add_edge(0, 1, weight=0.6)
    G.add_edge(0, 2, weight=0.2)
    G.add_edge(2, 3, weight=0.1)
    G.add_edge(2, 4, weight=0.7)
    G.add_edge(2, 5, weight=0.9)
    G.add_edge(0, 3, weight=0.3)

    # add node features
    G.nodes[0]["feat"] = [1]
    G.nodes[1]["feat"] = [2]
    G.nodes[2]["feat"] = [3]
    G.nodes[3]["feat"] = [4]
    G.nodes[4]["feat"] = [5]
    G.nodes[5]["feat"] = [6]

    return G

def draw_weighted_graph(G, node_features=None):

    # ---Visualization 
    pos = nx.spring_layout(G, seed=7) # positions for all nodes - seed for reproducibility
    nx.draw_networkx_nodes(G, pos, node_size=700) # nodes
    nx.draw_networkx_edges(G, pos, width=6) # edges

    # node feature
    if node_features is None:
        node_features = nx.get_node_attributes(G, "feat")
        print('get node featurex nx', node_features)
    nx.draw_networkx_labels(G, pos, labels=node_features) #font_size=20, font_family="sans-serif") # node labels
    
    # edge weight labels
    edge_weights = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_weights)

    ax = plt.gca() # to get the current axis
    ax.margins(0.08) # to avoid the nodes being clipped
    plt.axis("off") # to turn of the axis
    plt.tight_layout() # to make sure nothing gets clipped
    plt.show()

def compute_graph_convolution(graph, K=1, out_channels=2, normalization='sym'):
    # get graph dim
    edge_weights = torch.tensor([G[u][v]['weight'] for u, v in G.edges()], dtype=torch.float32) # (|E|, )
    print('weights', edge_weights.size())
    node_features = torch.tensor([G.nodes[i]['feat'] for i in G.nodes()], dtype=torch.float32) # (|V|, feat_dim)
    print('node_features', node_features.size())
    edge_indices = torch.tensor([e for e in G.edges()], dtype=torch.int64).permute(1, 0) # (2, |E|) 
    print('edge_indices', edge_indices.size())

    # compute graph convolution
    conv = ChebConv(
        in_channels=node_features.shape[1],
        out_channels=out_channels, K=K,
        normalization=normalization
        )

    out = conv(node_features, edge_indices, edge_weights) # (|V|, out_channels)
    print('out', out.size())
    return out


if __name__ == '__main__':
    G = create_simple_graph()
    #draw_weighted_graph(G)
    out = compute_graph_convolution(G, K=1, normalization='sym')
    node_features_after_conv = {i: torch.round(out[i]).tolist() for i in range(len(out))}
    print('node_features_after_conv', node_features_after_conv)
    draw_weighted_graph(G, node_features=node_features_after_conv)

    #node_feat = torch.tensor([v for v in node_features.values()])
    #node_features = torch.tensor([G.nodes[i]['feat'] for i in G.nodes()])
    #print(node_feat.size())
    #weights = torch.tensor([w for w in edge_weights.values()])
    #print(weights.size())
    

