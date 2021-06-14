from itertools import chain

import matplotlib.pyplot as plt
import networkx
from networkx.generators import complete_graph, circulant_graph

from minimum_feedback_vertex_set import mif


def recursive_dfs(g, n, ancestors: list, h: int, max_depth: int):
    solutions = []
    if max_depth == h:
        return tuple(sorted(ancestors))  # do sorted to avoid duplicates
    lst_neighbors = [x for x in g.neighbors(n) if x not in ancestors and not all(nn in ancestors for nn in g.neighbors(x))]
    if not lst_neighbors:
        lst_neighbors = set(g.nodes) - set(ancestors) - set([nn for a in ancestors for nn in g.neighbors(a)])
    for neigh in lst_neighbors:
            sol = recursive_dfs(g, neigh, ancestors + [neigh], h + 1, max_depth)
            if isinstance(sol, tuple):
                solutions.append(sol)
            else:
                solutions += sol
    return solutions


def find_solutions_dfs(dfs_tree, n, mif_size: int, ancestors=None):
    if ancestors is None:
        ancestors = []
    ancestors = ancestors + [n]
    if len(ancestors) == mif_size:
        return ancestors
    solutions = []
    for neigh in dfs_tree.neighbors(n):
        if neigh not in ancestors:
            sol = find_solutions_dfs(dfs_tree, neigh, mif_size, ancestors)
            if sol:
                solutions.append(tuple(sorted(sol)))
    return solutions


def find_other_mif(g, mif_size: int):
    solutions = set()
    for n in g.nodes:
        solutions.update(recursive_dfs(g, n, [], 0, mif_size))
    return solutions


def main():
    show = True
    # g = networkx.Graph([(i % n, (i + 1) % n) for i in range(n)])
    g = complete_graph(8)
    # g1 = networkx.Graph([(i+ 8 , (i + 1) % 4 + 8) for i in range(4)])
    # for e in g1.edges:
    #     g.add_edge(*e)
    if show:
        networkx.draw_networkx(g)
        plt.show()
    all_mif = find_all_mif(g)
    print('#solutions =', len(all_mif), 'all solutions', all_mif)


def find_all_mif(g):
    a_solution = mif(g.copy(), set(), t=None)
    all_solutions = find_other_mif(g, len(a_solution))
    return all_solutions


if __name__ == '__main__':
    main()
