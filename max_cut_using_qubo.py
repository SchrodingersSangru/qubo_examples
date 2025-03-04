import networkx as nx
from qubovert import QUBO
from qubovert.sim import anneal_qubo

# Create a graph
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

# Formulate QUBO
Q = QUBO()
for (u, v) in G.edges():
    Q[(u, u)] += 1
    Q[(v, v)] += 1
    Q[(u, v)] -= 2

# Solve using simulated annealing
result = anneal_qubo(Q, num_anneals=10)
cut = result.best.state
print(cut) 
