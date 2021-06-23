from itertools import combinations, chain
from typing import Set

import matplotlib.pyplot as plt
import networkx
from networkx.generators import complete_graph
from mif_github import MaximumInducedForest
from minimum_feedback_vertex_set import mif
from parser import parse, conditional_sample


def recursive_dfs(g, n, ancestors: list, max_depth: int):
    solutions = []
    if max_depth == len(ancestors):
        return tuple(sorted(ancestors))  # do sorted to avoid duplicates

    neighs = [x for x in g.neighbors(n) if
              x not in ancestors and not any(nn in ancestors for nn in g.neighbors(x) if nn != n)]
    if not neighs:
        neighs = find_new_node(ancestors, g)
    if not neighs:
        return solutions
    for neigh in neighs:
        sol = recursive_dfs(g, neigh, ancestors + [neigh], max_depth)
        if isinstance(sol, tuple):
            solutions.append(sol)
        else:
            solutions += sol
    return solutions


def find_new_node(ancestors, g):
    ancestors_set = set(ancestors)
    for x in set(g.nodes) - ancestors_set:
        sub_g_nodes = ancestors_set.union([x])
        sub_g = g.subgraph(sub_g_nodes)
        try:
            networkx.find_cycle(sub_g)
        except networkx.NetworkXNoCycle:
            return [x]


def find_other_mif(g, mif_size: int) -> Set[tuple]:
    solutions = set()
    for n in g.nodes:
        solutions.update(recursive_dfs(g, n, [n], mif_size))
    return solutions


def find_all_mif(g):
    """
    Find all Maximum Induced Forest i.e. the set of nodes such that G{MIF] doesn't contain a cycle
    # >>> g = complete_graph(20)
    # >>> g.remove_edge(0,3)
    # >>> solutions, cardinality = find_all_mif(g)
    # >>> cardinality
    # 3
    # >>> all(0 in sol and 3 in sol for sol in solutions)
    # True
    >>> g = networkx.Graph([(9,7),(7,8),(7,2),(7,0),(8,2),(0,2),(2,3),(2,6)])
    >>> solutions, cardinality = find_all_mif(g)
    >>> cardinality
    6
    >>> len(solutions)
    2
    >>> g = networkx.Graph([(9,5),(9,7),(7,3),(7,8),(7,5),(3,4),(3,5),(5,0),(5,2),(5,6),(8,5)])
    >>> networkx.draw_networkx(g); plt.show()
    >>> g.add_node(1)
    >>> solutions, cardinality = find_all_mif(g.copy())
    >>> cardinality
    9
    >>> len(solutions)
    2

    :param g: graph
    :return: a set of all the MIF and the cardinality of MIF got from exponential algorithm
    """
    a_solution = mif(networkx.MultiGraph(g.copy()), set(), t=None)
    all_solutions = find_other_mif(g, len(a_solution))
    return all_solutions, len(a_solution)


def find_all_minimal_fvs(g):
    mifs, cardinality = find_all_mif(g)
    min_fvs = {tuple(sorted(set(g.nodes).difference(mif))) for mif in mifs}
    return min_fvs, len(g.nodes) - cardinality


def find_all_solutions_fvs(g):
    mins_fvs, fvs_card = find_all_minimal_fvs(g)
    all_solutions = mins_fvs.copy()
    for min_fvs in mins_fvs:
        v_minus_min_fvs = set(g.nodes).difference(min_fvs)
        for k in range(1, len(v_minus_min_fvs)):
            for s in combinations(v_minus_min_fvs, k):
                all_solutions.add(tuple(sorted(chain(min_fvs, s))))
    all_solutions.add(tuple(sorted(g.nodes)))
    return all_solutions


def main():
    show = True
    # g = networkx.Graph([(i % n, (i + 1) % n) for i in range(n)])
    g = parse()
    g_nodes = conditional_sample(g, n=10, p=0.6)
    g = g.subgraph(g_nodes).copy()
    g = networkx.relabel_nodes(g, {n: i for i, n in enumerate(g.nodes)})
    print(type(g))
    if show:
        networkx.draw_networkx(g)
        plt.show()
    all_fvs = find_all_solutions_fvs(g)
    print('#solutions =', len(all_fvs), 'all solutions', all_fvs)


if __name__ == '__main__':
    main()
