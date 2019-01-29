# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 13:52:02 2019

@author: JamesChiou
"""
import sys

if len(sys.argv) > 2:
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
else:
    inputFile = "hw6_dataset.txt"
    outputFile = "output.txt"

MAX_NODE_NUMBER = 83000

graph = []
vertex_n = 0
for i in range(MAX_NODE_NUMBER):
    graph.append([])

with open(inputFile, "r") as file:
    for line in file:
        element = line.split()
        element = [int(i) for i in element]
        graph[element[0]].append(element[1])
        graph[element[1]].append(element[0])
        vertex_n = max(element[0], element[1], vertex_n)

degrees = dict((i, len(graph[i])) for i in range(vertex_n+1))
nodes = sorted(degrees, key=degrees.get)
bin_boundaries = [0]
curr_degree = 0
for i, v in enumerate(nodes):
    # i for index ; v for node number
    if degrees[v] > curr_degree:
        bin_boundaries.extend([i]*(degrees[v]-curr_degree))
        # find max degree
        curr_degree = degrees[v]

node_pos = dict((v, pos) for pos, v in enumerate(nodes))
# initial guesses
core = degrees
# find neighbors
nbrs = dict((v, set(graph[v])) for v in range(vertex_n+1))

for v in nodes:
    for u in nbrs[v]:
        # degree u > degree v
        if core[u] > core[v]:
            nbrs[u].remove(v)
            pos = node_pos[u]
            bin_start = bin_boundaries[core[u]]
            node_pos[u] = bin_start
            node_pos[nodes[bin_start]] = pos
            nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
            bin_boundaries[core[u]] += 1
            core[u] -= 1

k = max(core.values())  # max core
max_core_nodes = list(n for n in core if core[n] >= k)

with open(outputFile, "w") as file:
    for n in max_core_nodes:
        file.write(str(n) + '\n')
