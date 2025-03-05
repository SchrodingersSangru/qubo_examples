import networkx as nx
import random
from qubovert import QUBO
from qubovert.sim import anneal_qubo
from dwave.system import DWaveSampler
from dwave_neal import Neal
from dimod import BinaryQuadraticModel

# Create a random graph
num_nodes = random.randint(5, 10)
G = nx.gnm_random_graph(num_nodes, random.randint(num_nodes, num_nodes * 2))

# Formulate QUBO
Q = QUBO()
for (u, v) in G.edges():
    Q[(u, u)] += 1
    Q[(v, v)] += 1
    Q[(u, v)] -= 2


# Define your QUBO problem
bqm = BinaryQuadraticModel.from_qubo(Q)

# Create a Neal sampler
sampler = Neal()

# Sample the QUBO
sampleset = sampler.sample(bqm, num_reads=1000)

# Print the results
print(sampleset)
