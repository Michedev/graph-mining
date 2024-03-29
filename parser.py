import random
from dataclasses import dataclass
from itertools import chain
import networkx
import networkx as nx
from path import Path
import re
import json

from graph_stats import get_graph_stats

FIELD = 'formula'
DATA = Path(__file__).parent / 'sequences'
author_regex = re.compile(r'_([a-zA-Z\s\.\-]+)_')


def parse():
    authors = set()
    edges = []
    for file in DATA.files('*.json'):
        # print('opening', str(file))
        with open(file) as f:
            data_slice = json.load(f)
        for row in data_slice['results']:
            # print(row['author'])
            if 'author' in row:
                author_match = author_regex.match(row['author'])
                if author_match is not None:
                    author = author_match.group(1).lower()
                    authors.add(author)
                    if FIELD in row:
                        field_value = ''.join(row[FIELD])
                        for field_match in author_regex.finditer(field_value):
                            field_author = field_match.group(1).lower()
                            if author != field_author:
                                edges.append((author, field_author))
            else:
                print('skip', file, 'because key "author" is not present')
    num_map = dict()
    c = 0
    for n in chain(*edges):
        if n not in num_map:
            num_map[n] = c
            c += 1
    edges = [(num_map[u], num_map[v]) for u, v in edges]
    graph = networkx.Graph(edges)
    return graph


if __name__ == '__main__':
    graph = parse()
    print(get_graph_stats(graph))


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
            if chosen is None:  # fall back to random choice
                chosen = random.choice(remaining)
        remaining.remove(chosen)
        sample_nodes.append(chosen)
    return sample_nodes