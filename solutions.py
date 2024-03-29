import matplotlib.pyplot as plt
import networkx as nx
from parser import parse, conditional_sample
from minimum_feedback_vertex_set import minimum_feedback_vertex_set
from mif_github import MaximumInducedForest


def solution3(g=None):
    if g is None:
        g = parse()
        print('Graph parsed')
        sample_nodes = conditional_sample(g, 28, 0.5)
        g = g.subgraph(sample_nodes).copy()
    nx.draw_networkx(g)
    plt.savefig('input_graph.png')
    plt.close()
    g1 = g.copy()
    fig, axs = plt.subplots(ncols=2, figsize=(14, 9))
    sol1 = MaximumInducedForest().get_fbvs(g.copy())
    sol = minimum_feedback_vertex_set(g.copy())
    print('Minimum Feedback Vertex Set', sol)
    print('Github Minimum Feedback Vertex Set', sol1)

    for n in sol:
        g.remove_node(n)
    for n in sol1:
        g1.remove_node(n)
    nx.draw_networkx(g, ax=axs[0])
    nx.draw_networkx(g1, ax=axs[1])

    plt.savefig('output_graph.png')
    plt.close()

if __name__ == '__main__':
    solution3()
