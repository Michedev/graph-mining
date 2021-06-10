import random
import matplotlib.pyplot as plt
import networkx as nx
from parser import parse
from minimum_feedback_vertex_set import minimum_feedback_vertex_set

def conditional_sample(G: nx.Graph, n: int, p=0.3):
    sample_nodes = list()
    remaining = list(G.nodes)
    for _ in range(n):
        if random.random() > p:  # with probability (1 - p)
            chosen = random.choice(remaining)
        else:
            chosen = None
            for n in sample_nodes:
                neighs_n_remaining = set(G.neighbors(n)).intersection(remaining)
                neighs_n_remaining = list(neighs_n_remaining)
                if len(neighs_n_remaining) > 0:
                    chosen = random.choice(neighs_n_remaining)
            if chosen is None:  #fall back to random choice
                chosen = random.choice(remaining)
        remaining.remove(chosen)
        sample_nodes.append(chosen)
    return sample_nodes

def solution3(g=None):
    if g is None:
        g = parse()
        print('Graph parsed')
        sample_nodes = conditional_sample(g, 20, 0.8)
        g = g.subgraph(sample_nodes).copy()
    nx.draw_networkx(g)
    plt.savefig('input_graph.png')
    sol = minimum_feedback_vertex_set(g)
    print('Cardinality of Minimum Feedback Vertex Set', sol)

if __name__ == '__main__':
    solution3()