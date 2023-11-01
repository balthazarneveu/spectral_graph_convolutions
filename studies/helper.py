"""
Generic helpers to create latex reports and graph figures
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pprint
from typing import Callable, Optional

import functools
latex_mode = False


def load_graph(edge_path: Path) -> nx.Graph:
    """
    Load the network data into an undirected graph G
    using the read edgelist() function of NetworkX.
    """
    graph = nx.read_edgelist(edge_path)
    return graph

# ______________________________________________________________________________
# REPORT HELPERS
# ______________________________________________________________________________
def task(func: Callable):
    """Wrapper to split the results between tasks while printing
    When using the latex flag, it automatically adds the right latex
    language words:
    -Section name:
        - task_xx is deduced from the function name
        - description comes from the first line of the docstring
    - all prints will be translated to vertbatim so it looks like a command line log
    
    Author: Balthazar Neveu
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global latex_mode
        if latex_mode:
            sec_name = " ".join(func.__name__.split("_"))
            sec_name = sec_name.capitalize()
            sec_name += " : " + func.__doc__.split("\n")[0]
            print(r"\subsection*{%s}"%(sec_name))
            print(r"\begin{verbatim}")
            
        else:
            # Command line style
            print(40 * "-" + f" {func.__name__} " + 40 * "-")
            print(func.__doc__.split("\n")[0])
            print((len(func.__name__) + 2+ 80) * "_")
        results = func(*args, **kwargs)
        if latex_mode:
            print(r"\end{verbatim}")
        print("\n")
        return results
    return wrapper

def include_latex_figure(fig_name, legend, close_restart_verbatim=True, label=None):
    """Latex code to include a matplotlib generated figure"""
    fig_desc = [
        r"\end{verbatim}" if close_restart_verbatim else "",
        r"\begin{figure}[ht]",
        "\t"+r"\centering",
        "\t"+r"\includegraphics[width=.6\textwidth]{figures/%s}"%fig_name,
        "\t"+r"\caption{%s}"%legend,
        ("\t"+ r"\label{fig:%s}"%label) if label is not None else "",
        r"\end{figure}",
        r"\begin{verbatim}" if close_restart_verbatim else ""
    ]
    print("\n".join(fig_desc))
# ______________________________________________________________________________



def visualize_graph(
        ax: plt.Axes,
        graph: nx.Graph,
        title,
        color='lightgreen',
        properties=["degree"],
        node_labels=None,
        weights=False,
        seed=9):
    """Utility function to visualize a graph with a given title."""
    pos = nx.spring_layout(graph, seed=seed)
    # positions for all nodes - seed for reproducibility
    if node_labels is not None:
        color_map = [
            "red",
            "green",
            "blue",
            "purple",
            "lightgreen",
            "lightblue",
            
        ]
        color = [color_map[node_labels[node]%len(color_map)] for node in graph.nodes()]
    nx.draw(
        graph,
        pos,
        ax=ax,
        with_labels=True,
        node_size=700,
        node_color=color,
        
        font_size=15,
    )
    

    if weights:
        edge_labels = nx.get_edge_attributes(graph, "weight")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax)
    if "degree" in properties:
        degree_distribution = [f"{graph.degree(node):d}" for node in graph.nodes()]
        title += "\nDegree distribution:" + " ".join(degree_distribution)
    if "transitivity" in properties:
        transitivity = nx.transitivity(graph)
        title += f"\nTransitivity: {transitivity:.3f}"
    ax.set_title(title)

def create_graph_comparison(
        graph_def: list = [
            [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')],
            [('w', 'x'), ('x', 'y'), ('y', 'w'), ('z', 'z')]
        ],
        figure_folder=None,
        legend="",
        graph_names = ["Graph G1", "Graph G2"],
        colors=['lightblue', 'lightgreen'],
        fig_name="graph_comparison.png",
        properties=["degree"],
        node_labels:list=None,
        weights=False,
        seed=9,
        fig_width=5
    ):
    
    fig, axs = plt.subplots(1, len(graph_def), figsize=(len(graph_def)*fig_width, fig_width))
    for index, graph_x_def in enumerate(graph_def):
        if isinstance(graph_x_def, list):
            graph_x = nx.Graph()
            graph_x.add_edges_from(graph_x_def)
        elif isinstance(graph_x_def, nx.Graph):
            graph_x = graph_x_def
        visualize_graph(
            axs[index] if len(graph_def)>1 else axs,
            graph_x,
            graph_names[index],
            color=colors[index%2],
            properties=properties,
            weights=weights,
            node_labels=None if node_labels is None else node_labels[index],
            seed=seed
        )
    save_graph(
        figure_folder=figure_folder,
        fig_name=fig_name,
        legend=legend,
        close_restart_verbatim=False,
    )

    
def save_graph(figure_folder=None, fig_name=None, legend="", close_restart_verbatim=True):
    if figure_folder is not None:
        assert fig_name is not None
        fig_path = figure_folder/fig_name
        plt.savefig(fig_path)
        global latex_mode
        if not latex_mode:
            print(f"Saving {fig_path}")
        
        if latex_mode:
            include_latex_figure(
                fig_name,
                legend,
                close_restart_verbatim=close_restart_verbatim,
                label=fig_name.replace(".png", "")
            )
    else:
        plt.show()