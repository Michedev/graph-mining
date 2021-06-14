from itertools import combinations

import networkx
import matplotlib.pyplot as plt
from minimum_feedback_vertex_set import mif
from parser import parse, conditional_sample
from networkx.generators import circulant_graph, complete_graph
from networkx import bfs_tree, dfs_tree


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

def find_other_minimal_solutions(g, mif_size: int):
    solutions = list()
    for n in g.nodes:
        T = bfs_tree(g, n, depth_limit=mif_size)
        T1 = dfs_tree(g, n, mif_size)
        solutions += find_solutions_dfs(T, n, mif_size)
        solutions += find_solutions_dfs(T1, n, mif_size)
    return set(solutions)

def main():
    show = False
    n = 10
    g = networkx.Graph([(i%n, (i+1)%n) for i in range(n)])
    if show:
        networkx.draw_networkx(g)
        plt.show()
    a_solution = mif(g.copy(), set(), t=None)
    all_solutions = find_other_minimal_solutions(g, len(a_solution))
    # print('pairs', list(combinations(g.nodes, 2)))
    print('all solutions', all_solutions)

if __name__ == '__main__':
    main()