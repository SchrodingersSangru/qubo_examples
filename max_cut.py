import networkx as nx
import random
from neal import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel

# Create a random graph
num_nodes = random.randint(5, 10)
G = nx.gnm_random_graph(num_nodes, random.randint(num_nodes, num_nodes * 2))

# Formulate QUBO for Max-Cut
Q = {(i, i): 0 for i in range(num_nodes)}  # Initialize diagonal elements
for u, v in G.edges():
    Q[(u, v)] = 2  # Off-diagonal elements
    Q[(u, u)] -= 1  # Update diagonal elements
    Q[(v, v)] -= 1

# Convert to BinaryQuadraticModel
bqm = BinaryQuadraticModel.from_qubo(Q)

# Create a Simulated Annealing Sampler
sampler = SimulatedAnnealingSampler()

# Sample the QUBO (num_reads is the number of samples you want)
sampleset = sampler.sample(bqm, num_reads=1000)

# Print the best solution
print("Best solution found:")
solution = sampleset.first.sample

# Evaluate the solution
cut_value = sum(1 for u, v in G.edges() if sampleset.first.sample[u] != sampleset.first.sample[v])


best_solution = np.array([solution[i] for i in range(len(solution))])

print(f"Opimal output : {best_solution}")
print(f"Max-Cut value: {cut_value}")

# Print graph information
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {G.number_of_edges()}")
