import networkx as nx
import numpy as np
#from scipy.sparse import diags, eye
import matplotlib.pyplot as plt
from config import DEFAULT_FIGURES_LOCATION

import torch
from torch_geometric.nn import ChebConv

def compute_graph_laplacian(graph: nx.Graph) -> np.ndarray:
    """
    D^-1/2 W D^-1/2
    """

    W = nx.adjacency_matrix(graph).toarray() # (|V|, |V|)
    deg = np.sum(W, axis=1) 
    D = np.diag(np.sqrt(1./deg))

    return D@W@D

def create_simple_graph() -> nx.Graph:
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
    
    G.add_edge(0, 1, weight=.6)
    G.add_edge(0, 2, weight=.2)
    G.add_edge(2, 3, weight=.1)
    G.add_edge(2, 4, weight=0.7)
    G.add_edge(2, 5, weight=0.9)
    G.add_edge(0, 3, weight=0.3)

    # add node features
    G.nodes[0]["feat"] = [4]
    G.nodes[1]["feat"] = [5]
    G.nodes[2]["feat"] = [2]
    G.nodes[3]["feat"] = [2]
    G.nodes[4]["feat"] = [3]
    G.nodes[5]["feat"] = [3]

    return G

def draw_weighted_graph(G: nx.Graph, node_features: dict = None, fname: str = None):
    fig = plt.figure() # figsize=(8, 8)

    # ---Visualization 
    pos = nx.spring_layout(G, seed=7) # positions for all nodes - seed for reproducibility
    nx.draw_networkx_nodes(G, pos, node_size=700) # nodes
    nx.draw_networkx_edges(G, pos, width=6) # edges

    # node feature
    if node_features is None:
        node_features = nx.get_node_attributes(G, "feat")
    print('node_features', node_features)
    nx.draw_networkx_labels(G, pos, labels=node_features) #font_size=20, font_family="sans-serif") # node labels
    
    # edge weight labels
    edge_weights = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_weights)

    ax = plt.gca() # to get the current axis
    ax.margins(0.08) # to avoid the nodes being clipped
    plt.axis("off") # to turn of the axis
    plt.tight_layout() # to make sure nothing gets clipped
    fig.savefig(fname, bbox_inches='tight') # bbox_inches='tight' to avoid the labels being clipped


def compute_graph_convolution(G: nx.Graph, K: int =1, out_channels: int =2, normalization: str ='sym', initialization: str =None):
    """
    Parameters:
    ----------
    G: networkx graph
    """

    # get graph dim
    edge_weights = torch.tensor([G[u][v]['weight'] for u, v in G.edges()], dtype=torch.float32) # (|E|, )
    node_features = torch.tensor([G.nodes[i]['feat'] for i in G.nodes()], dtype=torch.float32) # (|V|, feat_dim)
    edge_indices = torch.tensor([e for e in G.edges()], dtype=torch.int64).permute(1, 0) # (2, |E|) 

    # compute graph convolution
    conv = ChebConv(
        in_channels=node_features.shape[1],
        out_channels=out_channels, K=K,
        normalization=normalization
        )
    
    # initialize weights
    if initialization is not None:
        init = eval(f"torch.nn.init.{initialization}_")
        for lins in conv.lins:
            init(lins.weight)

    out = conv(node_features, edge_indices, edge_weights) # (|V|, out_channels)
    return out


def matprint(mat, fmt=".03f"):
    # Inspired from https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    mat_str = ""
    for x in mat:
        mat_row = "[ "
        for i, y in enumerate(x):
            digit = ("{:"+str(col_maxes[i])+fmt+"}").format(y)
            if y == 0:
                digit = " "*((len(digit)-1 )//2 )+  "0" + " "*((len(digit)-1)//2)
            mat_row+= digit + " "
        mat_row+= "  ]"
        mat_str+= mat_row + "\n"
    mat_str = mat_str[:-1]
    print(mat_str)
    return mat_str

if __name__ == '__main__':
    K=3
    G = create_simple_graph()
    draw_weighted_graph(G, fname=f"{DEFAULT_FIGURES_LOCATION}/toy_graph_init.png")
    # print graph
    norm_lap = compute_graph_laplacian(G)
    print('normalized laplacian matrix')
    matprint(norm_lap)
    print('normalized laplacian matrix²')
    matprint(norm_lap@norm_lap)
    print('normalized laplacian matrix³')
    matprint(norm_lap@norm_lap@norm_lap)

    out_conv = compute_graph_convolution(G, K=K, out_channels=1, normalization='sym', initialization="ones")
    node_features_after_conv = {i: torch.round(out_conv[i]).tolist() for i in range(len(out_conv))}
    print('node_features_after_conv', node_features_after_conv)
    draw_weighted_graph(G, node_features=node_features_after_conv, fname=f'{DEFAULT_FIGURES_LOCATION}/toy_graph_conv_K{K}.png')

    

