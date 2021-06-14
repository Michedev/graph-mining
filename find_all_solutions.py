from itertools import chain

import matplotlib.pyplot as plt
import networkx
from networkx.generators import complete_graph
from mif_github import MaximumInducedForest
from minimum_feedback_vertex_set import mif


def recursive_dfs(g, n, ancestors: list, h: int, max_depth: int, ccs: list):
    solutions = []
    if max_depth == h:
        return tuple(sorted(ancestors))  # do sorted to avoid duplicates
    lst_neighbors = [x for x in g.neighbors(n) if x not in ancestors and not any(nn in ancestors for nn in g.neighbors(x) if nn != n)]
    if not lst_neighbors:
        ccs = [cc for cc in ccs if not any(x in cc for x in ancestors)]
        lst_neighbors = chain(*ccs)
    for neigh in lst_neighbors:
            sol = recursive_dfs(g, neigh, ancestors + [neigh], h + 1, max_depth, ccs)
            if isinstance(sol, tuple):
                solutions.append(sol)
            else:
                solutions += sol
    return solutions


def find_other_mif(g, mif_size: int):
    solutions = set()
    ccs = list(networkx.connected_components(g))
    for n in g.nodes:
        solutions.update(recursive_dfs(g, n, [], 0, mif_size, ccs))
    return solutions


def main():
    show = True
    # g = networkx.Graph([(i % n, (i + 1) % n) for i in range(n)])
    g = complete_graph(9)
    n = len(g.nodes)
    g1 = networkx.Graph([(i+ n , (i + 1) % 4 + n) for i in range(4)])
    for e in g1.edges:
        g.add_edge(*e)
    if show:
        networkx.draw_networkx(g)
        plt.show()
    all_mif, card_solution = find_all_mif(g)
    print('cardinality', card_solution)
    print('#solutions =', len(all_mif), 'all solutions', all_mif)


def find_all_mif(g):
    """
    Find all Maximum Induced Forest i.e. the set of nodes such that G{MIF] doesn't contain a cycle
    >>> g = complete_graph(20)
    >>> g.remove_edge(0,3)
    >>> solutions, cardinality = find_all_mif(g)
    >>> cardinality
    3
    >>> all(0 in sol and 3 in sol for sol in solutions)
    True

    :param g: graph
    :return: a set of all the MIF and the cardinality of MIF got from exponential algorithm
    """
    a_solution = mif(networkx.MultiGraph(g.copy()), set(), t=None)
    all_solutions = find_other_mif(g, len(a_solution))
    return all_solutions, len(a_solution)


if __name__ == '__main__':
    main()
