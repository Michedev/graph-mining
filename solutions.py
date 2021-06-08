import random

from parser import parse
from minimum_feedback_vertex_set import minimum_feedback_vertex_set


def solution3(g=None):
    if g is None:
        g = parse()
        print('Graph parsed')
        sample_nodes = random.sample(g.nodes, int(len(g.nodes) * 0.25))
        g = g.subgraph(sample_nodes)
    sol = minimum_feedback_vertex_set(g)
    print('Cardinality of Minimum Feedback Vertex Set', sol)

if __name__ == '__main__':
    solution3()