import networkx as nx
import matplotlib.pyplot as plt

def create_simple_graph(nbNodes=5):
    G = nx.Graph()
    G.add_nodes_from(range(nbNodes))
    for i in range(1, nbNodes):
        G.add_edge(0, i)

    return G

if __name__ == '__main__':
    G = create_simple_graph()
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()
    print("Done")