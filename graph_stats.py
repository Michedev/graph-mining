import networkx


def get_graph_stats(g: networkx.Graph):
    m = len(g.edges)
    n = len(g.nodes)
    max_degree = max(len(list(g.neighbors(node))) for node in g.nodes)
    edge_density = m / (n * (n-1) / 2)
    return dict(n=n, m=m, max_degree=max_degree, edge_density=edge_density)