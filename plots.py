from itertools import combinations
from path import Path
import matplotlib.pyplot as plt
import networkx as nx
from minimum_feedback_vertex_set import minimum_feedback_vertex_set

PLOT = Path(__file__).parent / 'plots'
if not PLOT.exists():
    PLOT.mkdir()



def plot_minimum_feedback_vertex_set():
    n = 10
    g = nx.Graph([(i % n, (i + 1) % n) for i in range(n)])
    sol = minimum_feedback_vertex_set(g)
    g_complete = g.copy()
    for u, v in combinations(g.nodes, 2):
        if u != v and not g.has_edge(u, v):
            g_complete.add_edge(u, v)
    sol_complete = minimum_feedback_vertex_set(g_complete)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 6))
    axs[0].set_title('Input Graph')
    axs[1].set_title('Output Graph')
    nx.draw_networkx(g, ax=axs[0])
    for n in sol:
        g.remove_node(n)
    nx.draw_networkx(g, ax=axs[1])
    plt.savefig(PLOT / 'cycle_graph.png')
    plt.close()
    fig, axs = plt.subplots(ncols=2, figsize=(10, 6))
    axs[0].set_title('Input Graph')
    axs[1].set_title('Output Graph')
    nx.draw_networkx(g_complete, ax=axs[0])
    for n in sol_complete:
        g_complete.remove_node(n)
    nx.draw_networkx(g_complete, ax=axs[1])
    plt.savefig(PLOT / 'complete_graph.png')
    plt.close()


if __name__ == '__main__':
    plot_minimum_feedback_vertex_set()
