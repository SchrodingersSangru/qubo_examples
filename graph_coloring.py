import dimod
import networkx as nx
import matplotlib.pyplot as plt
from neal import SimulatedAnnealingSampler


# Define the graph

num_nodes = 10 # user input number of nodes  to the graph 
G = nx.cycle_graph(num_nodes)  # Example graph: a cycle with 5 nodes

# Define the number of colors
num_colors = 3

# Create a Binary Quadratic Model (BQM)
bqm = dimod.BinaryQuadraticModel(dimod.BINARY)

# Add constraints for each node to have exactly one color
for node in G.nodes:
    variables = [f"{node}_{color}" for color in range(num_colors)]
    bqm.add_linear_equality_constraint(
        [(var, 1) for var in variables],
        lagrange_multiplier=10,  # Penalty weight
        constant=-1,
    )

# Add constraints to ensure no two adjacent nodes share the same color
for u, v in G.edges:
    for color in range(num_colors):
        bqm.add_interaction(f"{u}_{color}", f"{v}_{color}", 10)  # Penalty weight

# Solve the BQM using the Neal sampler (simulated annealing)

sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=100)

# Extract the best solution
best_sample = sampleset.first.sample

# Map solution to node colors
node_colors = {}
for node in G.nodes:
    for color in range(num_colors):
        if best_sample[f"{node}_{color}"] == 1:
            node_colors[node] = color

# Visualize the graph with colored nodes
nx.draw(
    G,
    with_labels=True,
    node_color=[node_colors[node] for node in G.nodes],
    cmap=plt.cm.rainbow,
    node_size=500,
)
plt.show()
