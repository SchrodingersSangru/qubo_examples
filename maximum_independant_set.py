import networkx as nx
import neal
import matplotlib.pyplot as plt
import random

# Create a random graph
num_nodes = 10
edge_probability = 0.3

G = nx.gnp_random_graph(num_nodes, edge_probability)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=16, font_weight='bold')
plt.title("Random Graph")
plt.show()

# Create QUBO for Maximum Independent Set
Q = {(node, node): -1 for node in G.nodes}
Q.update({(u, v): 2 for u, v in G.edges})

# Create a simulated annealing sampler
sampler = neal.SimulatedAnnealingSampler()

# Sample from the QUBO
sampleset = sampler.sample_qubo(Q, num_reads=1000)

# Get the best solution
sample = sampleset.first.sample

# Extract the independent set
independent_set = [node for node in G.nodes if sample[node] > 0]

print(f"Maximum Independent Set: {independent_set}")
print(f"Size of Maximum Independent Set: {len(independent_set)}")

# Verify the solution
is_independent = all(not G.has_edge(u, v) for u in independent_set for v in independent_set if u != v)
print(f"Is a valid independent set: {is_independent}")

# Highlight the independent set in the graph
node_colors = ['red' if node in independent_set else 'lightblue' for node in G.nodes]
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=16, font_weight='bold')
plt.title("Random Graph with Maximum Independent Set")
plt.show()
