import networkx as nx
import numpy as np
#from scipy.sparse import diags, eye
import matplotlib.pyplot as plt
from config import DEFAULT_FIGURES_LOCATION
from PIL import Image, ImageDraw, ImageFont
from torch_geometric.utils import from_networkx

import torch
from torch_geometric.nn import ChebConv

def compute_graph_laplacian(graph: nx.Graph, normalization="sym") -> np.ndarray:
    """Networkx graph to Laplacian matrix.
    Assumes undirected graphs!
    
    - normalization = None -> D -W
    - normalization = `sym` ->  Id - D^-1/2 W D^-1/2
    """
    W = nx.adjacency_matrix(graph).toarray() # (|V|, |V|)
    deg = np.sum(W, axis=1)
    if normalization is None:
        return np.diag(deg)-W
    D = np.diag(np.sqrt(1./deg))

    return np.eye(W.shape[0])-D@W@D

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
        node_features = {idx: f"{idx:d}" for idx, val in node_features.items()}
    # print('node_features', node_features)
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
    g = from_networkx(G)

    # get graph dim
    edge_weights = g.weight.type(torch.float32) # torch.float32 (|E|, ) 
    node_features = g.feat.type(torch.float32) # torch.float32 (|V|, feat_dim)
    edge_indices = g.edge_index.type(torch.long) # torch.int64 (2, |E|)

    # compute graph convolution
    conv = ChebConv(
        in_channels=node_features.shape[1],
        out_channels=out_channels,
        K=K,
        normalization=normalization
    )

    # initialize weights
    if initialization is not None:
        init = eval(f"torch.nn.init.{initialization}_")
        for lins in conv.lins:
            init(lins.weight)
    # Checking every weights is 1
    # for idx in range(len(conv.lins)):
    #     print(f"{idx} ->{conv.lins[idx].weight}")
    out = conv(node_features, edge_indices, edge_weights) # (|V|, out_channels)
    return out


def matprint(mat, fmt=".02f"):
    # Inspired from https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    mat_str = ""
    for x in mat:
        mat_row = "[ "
        for i, y in enumerate(x):
            digit = ("{:"+str(col_maxes[i])+fmt+"}").format(y)
            if y == 0 or y==1 or y==-1:
                digit = " "*((len(digit)-1 )//2 )+  f"{int(y):d}" + " "*((len(digit)-1)//2)
            mat_row+= digit + " "
        mat_row+= " ]"
        mat_str+= mat_row + "\n"
    mat_str = mat_str[:-1]
    print(mat_str)
    return mat_str



def save_string_to_image(text: str, fname: str, size=(400, 400)):
    # Create an image with white background and add black text on it
    img = Image.new('RGB', size, color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("cour.ttf", 16)
    text_position = (10, 50)
    d.text(text_position, text, fill=(0, 0, 0), font=font)
    img.save(fname)


def main():
    K=3
    G = create_simple_graph()
    draw_weighted_graph(G, fname=f"{DEFAULT_FIGURES_LOCATION}/toy_graph_init.png")
    
    # print graph
    norm_lap = compute_graph_laplacian(G)
    print('normalized laplacian matrix')
    save_string_to_image(matprint(norm_lap), fname=f"{DEFAULT_FIGURES_LOCATION}/lap1.png")
    print('normalized laplacian matrix²')
    save_string_to_image(matprint(norm_lap@norm_lap), fname=f"{DEFAULT_FIGURES_LOCATION}/lap2.png")
    print('normalized laplacian matrix³')
    save_string_to_image(matprint(norm_lap@norm_lap@norm_lap), fname=f"{DEFAULT_FIGURES_LOCATION}/lap3.png")

    node_features = np.array([G.nodes[i]['feat'] for i in G.nodes()], dtype=np.float32) # (|V|, feat_dim)
    

    for k in range(1, K+1):
        print(20*"-" + f" {k=} " + 20*"-")
        out_conv = compute_graph_convolution(G, K=k, out_channels=1, normalization='sym', initialization="ones")
        node_features_after_conv = {i: f"{out_conv[i][0]:.2f}" for i in range(len(out_conv))}
        print('node_features_after_conv', node_features_after_conv)
        
        id = np.eye(norm_lap.shape[0])
        t0 = node_features
        t1 = np.dot(norm_lap-id, node_features)
        if k>1:
            print("Manual ChebConv - (weights=1)")
        if k==2:
            matprint(t0+t1)
        if k==3:
            t2 = 2* np.dot(norm_lap-id, t1) - t0
            matprint(t2+t1+t0)
        print("ChebConv - (weights=1) results")
        matprint(out_conv.detach().numpy())
        
        draw_weighted_graph(G, node_features=node_features_after_conv, fname=f'{DEFAULT_FIGURES_LOCATION}/toy_graph_conv_K{k}.png')


        # NOTE!
        # Doc is deceptive: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ChebConv
        # "Message passing" propagation seems to mean applying (L~  - Id)

        # Forward in cheb_conv
        # Tx_0 = x
        # Tx_1 = x  # Dummy.
        # out = self.lins[0](Tx_0)

        # # propagate_type: (x: Tensor, norm: Tensor)
        # if len(self.lins) > 1:
        #     Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
        #     out = out + self.lins[1](Tx_1)

        # for lin in self.lins[2:]:
        #     Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
        #     Tx_2 = 2. * Tx_2 - Tx_0
        #     out = out + lin.forward(Tx_2)
        #     Tx_0, Tx_1 = Tx_1, Tx_2


if __name__ == '__main__':
    main()
    

    

