from dataclasses import dataclass

import networkx
from path import Path
import re
import json

FIELD = 'formula'
DATA = Path(__file__).parent / 'sequences'
author_regex = re.compile(r'_([a-zA-Z\s\.\-]+)_')


def run():
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
                            edges.append((author, field_author))
    graph = networkx.Graph(edges)
    return graph


if __name__ == '__main__':
    graph = run()
    print(graph.edges)
