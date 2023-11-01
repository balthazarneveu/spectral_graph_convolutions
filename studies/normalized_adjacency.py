"""
Compute graph normalized ajacency matrix
"""

import networkx as nx
import numpy as np
from scipy.sparse import diags, eye
from helper import create_graph_comparison, task
from pathlib import Path
import helper



def compute_normalized_adjacency(graph: nx.Graph) -> np.ndarray:
    """D^-1/2 . A . D^-1/2
    This is not simple averaging in a neighborhood.
    It is a weighted average, where each weight is inversely proportional to the sqrt(degree of the node).
    Rows do not sum to 1. This smoothing operation does not conserve energy.
    """
    node_list = graph.nodes()
    adj = nx.adjacency_matrix(graph, nodelist=node_list)
    
    A = adj + eye(adj.shape[0])
    deg = np.sum(A, axis=1) # This matches the definition of the degree
    print(deg)
    d = np.sqrt(1./deg)
    D = diags(d)
    nAdj = D.dot(A).dot(D) # keeps the spasity!
    return nAdj


def compute_averaging(graph: nx.Graph) -> np.ndarray:
    """D^-1.A
    This is simple averaging in a neighborhood
    Equivalent to a good old box filter in a regaular image grid
    """
    node_list = graph.nodes()
    adj = nx.adjacency_matrix(graph, nodelist=node_list)
    
    A = adj + eye(adj.shape[0])
    deg = np.sum(A, axis=1) # This matches the definition of the degree

    d = 1./deg
    D = diags(d)
    nAdj = D.dot(A) # keeps the spasity!
    return nAdj

def complex_graph_example(figure_folder=None):
    """Star example"""
    n1 = 4
    n2 = 4
    n_star = 8
    G1 = nx.complete_graph(n1)
    G2 = nx.cycle_graph(n2)
    G3 = nx.star_graph(n_star-1)
    # Relabel nodes of G2 to ensure they are disjoint from G1
    G2 = nx.relabel_nodes(G2, {i: i + n1 for i in range(n2)})
    # Relabel nodes of G3 to ensure they are disjoint from G1 and G2
    G3 = nx.relabel_nodes(G3, {i: i + n1+n2 for i in range(n_star)})

    # Combine the two graphs
    G = nx.union(G1, G2)
    # Combine the three graphs
    G = nx.union(G, G3)

    # G.add_edge(0, n1)
    # G.add_edge(n1, n2)
    G.add_edge(n1+n2//2, n1//2)
    G.add_edge(n1+n2, 1)
    # nodel_labels = spectral_clustering(G, 3, d=3, sparse=False, debug_prints=True)

@task
def star_example(figure_folder=None):
    """Analyzis of the normalized adjacency matrix for the star graph example"""
    n_star = 8
    G = nx.star_graph(n_star-1)
    # G = nx.star_graph(50-1)
    DAD = compute_normalized_adjacency(G)
    # DAD = compute_averaging(G)
    print(DAD.toarray())

    for i in range(DAD.shape[0]):
        if i!=0:
            continue
        for j in range(DAD.shape[1]):
            if DAD[i,j] > 0:
                G.add_weighted_edges_from([
                    (i,
                     j, 
                    #  j if i!=j else f"{j}",
                    f"{DAD[i,j]:0.2f}")
                ])

    create_graph_comparison(
        [G],
        # node_labels=[nodel_labels],
        properties=[],
        weights=True,
        figure_folder=figure_folder,
        fig_name="adjacency_smoothing_star.png",
        legend="Feature smoothing using the normalized adjacency matrix - Edges indicate the weights for the central node convolution.",
        graph_names=["Feature smoothing using the normalized adjacency matrix\n - Star graph\nWeights for the central node"],
        seed=42,
        fig_width=10
    )


def lin2coord(i, W):
    x = i//W
    y = i - x*W
    return (x, y)

@task
def grid_check(figure_folder=None):
    """Analyzis of the normalized adjacency matrix for the regular grid graph example"""
    W, H = 5, 7
    G = nx.grid_2d_graph(W, H)
    DAD = compute_normalized_adjacency(G)
    DAD = DAD.toarray()
    print(DAD)
    # G.add_weighted_edges_from([((0, 0), (0, 1), 2)])
    for i in range(DAD.shape[0]):
        if i!=DAD.shape[0]//2: # PICK CENTRAL NODE ONLY
            continue
        # if i!=3:
        #     continue
        for j in range(DAD.shape[1]):
            if DAD[i,j] > 0:
                # print(i, j, DAD[i,j])
                G.add_weighted_edges_from([
                    (lin2coord(i, H),
                     lin2coord(j, H),
                     f"{DAD[i,j]:0.2f}")
                ])
    create_graph_comparison(
        [G],
        # node_labels=[nodel_labels],
        properties=[],
        weights=True,
        figure_folder=figure_folder,
        fig_name="adjacency_smoothing_regular_grid.png",
        legend="Feature smoothing using the normalized adjacency matrix",
        graph_names=["Feature smoothing using the normalized adjacency matrix\n - Regular grid\nWeights for the central node"],
        seed=42,
        fig_width=10
    )
if __name__ == '__main__':
    helper.latex_mode = True
    figures_folder = Path(__file__).parent/"custom_figures"
    figures_folder.mkdir(parents=True, exist_ok=True)
    # figures_folder=None
    grid_check(figure_folder=figures_folder)
    star_example(figure_folder=figures_folder)