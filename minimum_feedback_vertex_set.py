import random
from operator import itemgetter
from typing import Union, Set
from itertools import combinations
import matplotlib.pyplot as plt
import networkx

N = Union[str, int]


def id_(G: networkx.Graph, T: Set[N], t: N):
    """
    This function represent the operation "Id" described by the paper as "the operation of contracting all edges of T
    into one vertex t and removing appeared loops."
    Paper link: https://www.cse.unsw.edu.au/~sergeg/papers/Iwpec2006.pdf
    :return: the new graph
    """
    print(T)
    print('t', t)
    G1 = G
    G1_T: networkx.Graph = G1.subgraph(T)
    while len(G1_T.nodes) > 1:
        e_gen = filter(lambda e: t in e, G1_T.edges)
        try:
            e = next(e_gen)
            if e[0] != t:
                e = e[1], e[0]
        except StopIteration:
            e = None
        if e is not None:
            G1 = networkx.algorithms.contracted_edge(G1, e, self_loops=False)
            T.discard(e[1])
            print('discarded', e[1])
            G1_T = G1.subgraph(T)
    return G1


def id_star(G: networkx.Graph, T: Set[N], t: N):
    """
    This function represent the operation "Id*" described by the paper as "the operation Id(T, t)
    followed by the removal of all vertices connected with t by multiedges."
    Paper link: https://www.cse.unsw.edu.au/~sergeg/papers/Iwpec2006.pdf
    :return: the new graph
    """

    G1 = id_(G, T, t)
    G1_t: networkx.Graph = G1.subgraph([t] + list(G1.neighbors(t)))
    found_edges = []
    for e in G1_t.edges:
        e_sorted = sorted(e)
        if e_sorted in found_edges:  # it means it's a duplicate
            print('removed edge', e, "because it's a duplicate")
            G1.remove_edge(*e)
        else:
            found_edges.append(e_sorted)
    return G1


def get_max_degree(G: networkx.Graph) -> int:
    return max(map(itemgetter(1), G.degree()))


def build_graph_proposition_3(G, F, t, neighbors_t):
    G_n_t: networkx.Graph = G.subgraph(neighbors_t)
    for (u, v) in combinations(G_n_t.nodes, 2):
        if not G_n_t.has_edge(u, v):
            neigh_u = set(G_n_t.neighbors(u))
            neigh_v = set(G_n_t.neighbors(v))
            if len(neigh_v.intersection(neigh_u)) >= 1:  # i.e. have at least a common neighbor
                G_n_t.add_edge(u, v)
    return G_n_t


def find_cycle_nodes(G, n, size):
    for cycle in networkx.find_cycle(G, n):
        if len(cycle) == size:
            nodes_triangle = set()
            for e in cycle:
                nodes_triangle.add(e[0])
                nodes_triangle.add(e[1])
            nodes_triangle.remove(n)
            return nodes_triangle
    raise Exception("Triangle not found")


def find_triangle_nodes(G, n):
    return find_cycle_nodes(G, n, 3)


def find_square_nodes(G, n):
    return find_cycle_nodes(G, n, 4)


def is_indipendent(G: networkx.Graph, s: Set[N]):
    for (u, v) in combinations(s, 2):
        if G.has_edge(u, v):
            return False
    return True


def mif(G: networkx.Graph, F: Set[N]):
    """
    Calculate the size of the Maximal Indipendent Forest using SOTA algorithm
    with complexity O(1.7548^n) where n is the number of nodes of the graph
    Paper link: https://www.cse.unsw.edu.au/~sergeg/papers/Iwpec2006.pdf
    Example:

    >>> n = 10
    >>> g = networkx.Graph([(i%n, (i+1)%n) for i in range(n)])
    >>> assert mif(g, set()) == 9


    :param G: graph
    :param F: The forest i.e. a set of nodes used for recursive calls. If you want to find the maximum indipendent forest just pass set()
    :return: the number of nodes for maxmimum indipendent forest
    """
    ccs_nodes = list(networkx.connected_components(G))  # preprocessing 1
    if len(ccs_nodes) > 1:
        tot = 0
        for cc_nodes in ccs_nodes:
            cc = G.subgraph(cc_nodes)
            F_i = F.intersection(cc_nodes)
            tot += mif(cc, F_i)
        print('returning', tot)
        return tot
    if not is_indipendent(G, F):  # preprocessing 2
        G1 = G.copy()
        for cc_nodes in networkx.connected_components(G1.subgraph(F)):
            if cc_nodes and len(cc_nodes) > 1:  # i.e. is not trivial
                v_t = random.choice(list(cc_nodes))
                G1 = id_star(G1, cc_nodes, v_t)
        F1 = set(G1.nodes).intersection(F)
        tot = len(F.difference(F1))
        tot += mif(G1, F1)
        print('p2 - returning', tot)
        return tot
    if len(F) == len(G.nodes):  # case 1
        print('case 1')
        print('returning', len(F))
        return len(F)
    max_degree = get_max_degree(G)
    F_empty = len(F) == 0
    if F_empty and max_degree <= 1:  # case 2
        print('case 2')
        print('returning', len(G.nodes))
        return len(G.nodes)
    elif F_empty and max_degree >= 2:  # case 3
        v = next(n for n in G.nodes if G.degree[n] >= 2)
        F1 = F.copy()
        F1.add(v)
        G1 = G.copy()
        G1.remove_node(v)
        print('case 3')
        return max(mif(G, F1), mif(G1, F))
    t = random.choice(list(F))  # set active vertex
    V_wo_F = set(G.nodes).difference(F)
    neighbors_t = set(G.neighbors(t))
    if V_wo_F == neighbors_t:
        G_p3 = build_graph_proposition_3(G, F, t, neighbors_t)
        print(G_p3.nodes)
        I = networkx.maximal_independent_set(G_p3)
        print('case 5')
        print('returning', len(I) + len(F))
        return len(I) + len(F)
    neighbors_t = list(G.neighbors(t))
    gen_degree_neight_t = networkx.generalized_degree(G, neighbors_t)
    neighs_gen_degree_less_1 = []
    neighs_gen_degree_gr_4 = []
    neighs_gen_degree_eq_2 = []
    for neighbor, gen_degree in gen_degree_neight_t.items():
        if gen_degree[3] <= 1:
            neighs_gen_degree_less_1.append(neighbor)
        if gen_degree[3] >= 4:
            neighs_gen_degree_gr_4.append(neighbor)
        if gen_degree[3] == 2:
            neighs_gen_degree_eq_2.append(neighbor)
    neigh_gen_degree_less_1 = random.choice(neighs_gen_degree_less_1) if neighs_gen_degree_less_1 else None
    neigh_gen_degree_gr_4 = random.choice(neighs_gen_degree_gr_4) if neighs_gen_degree_gr_4 else None
    neigh_gen_degree_eq_2 = random.choice(neighs_gen_degree_eq_2) if neighs_gen_degree_eq_2 else None
    if neigh_gen_degree_less_1 is not None:  # case 6
        F1 = F.copy()
        F1.add(neigh_gen_degree_less_1)
        print('case 6')
        return mif(G, F1)
    del neigh_gen_degree_less_1
    if neigh_gen_degree_gr_4 is not None:  # case 7
        F1 = F.copy()
        F1.add(neigh_gen_degree_gr_4)
        G1 = G.copy()
        G1.remove_node(neigh_gen_degree_gr_4)
        F2 = F.copy()
        if neigh_gen_degree_gr_4 in F2:
            F2.remove(neigh_gen_degree_gr_4)
        print('case 7')
        return max(mif(G, F1), mif(G1, F2))
    del neigh_gen_degree_gr_4
    if neigh_gen_degree_eq_2:  # case 8
        nodes_triangle = find_triangle_nodes(G, neigh_gen_degree_eq_2)
        F1 = F.copy()
        F1.add(neigh_gen_degree_eq_2)
        G1 = G.copy()
        G1.remove_node(neigh_gen_degree_eq_2)
        F2 = F.copy()
        F2.union(nodes_triangle)
        return max(mif(G, F1), mif(G1, F2))
    v = find_v_case_9(G, neighbors_t)  # case 9
    other_nodes = list(find_square_nodes(G, v))
    F1 = F.copy();
    F1.add(v)
    G1 = G.copy()
    G1.remove_node(v)
    w1 = other_nodes[0]
    w2, w3 = other_nodes[1:]
    F2 = F.copy();
    F2.add(w1)
    G2 = G.copy();
    G2.subgraph([v, w1])
    F3 = F.copy();
    F3.add(w2)
    F3.add(w3)
    return max(mif(G, F1), mif(G1, F2), mif(G2, F3))


def find_v_case_9(G, neighbors_t):
    for neighbor_t in neighbors_t:  # case 9
        for neighbor_neighbor_t in G.neighbors(neighbor_t):
            if neighbor_neighbor_t not in neighbors_t:
                return neighbor_neighbor_t
    raise Exception("V not found")


def minimum_feedback_vertex_set(g: networkx.Graph):
    """
    Calculate the size of the minimum feedback vertex set using SOTA algorithm
    with complexity O(1.7548^n) where n is the number of nodes of the graph
    Paper link: https://www.cse.unsw.edu.au/~sergeg/papers/Iwpec2006.pdf

    Example:

    >>> n = 10
    >>> g = networkx.Graph([(i%n, (i+1)%n) for i in range(n)])
    >>> assert minimum_feedback_vertex_set(g, set()) == 1

    :param G: graph
    :return: the cardinality of minimal feedback vertex set
    """
    return len(g.nodes) - mif(g, set())


if __name__ == '__main__':
    n = 10
    g = networkx.Graph([(i % n, (i + 1) % n) for i in range(n)])
    print(minimum_feedback_vertex_set(g))
