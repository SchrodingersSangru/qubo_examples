import networkx as nx
import random
from dwave_neal import Neal
from dimod import BinaryQuadraticModel

# Generate a random graph
num_nodes = random.randint(5, 10)
num_edges = random.randint(num_nodes, num_nodes * 2)
G = nx.gnm_random_graph(num_nodes, num_edges)

# Generate random sets (subgraphs)
num_sets = random.randint(3, 6)
sets = {}
for i in range(num_sets):
    set_name = chr(65 + i)  # A, B, C, ...
    set_nodes = random.sample(list(G.nodes()), random.randint(2, num_nodes-1))
    set_cost = random.randint(1, 5)
    sets[set_name] = {'items': set_nodes, 'cost': set_cost}

# Formulate QUBO
Q = {}
for set_name in sets:
    Q[(set_name, set_name)] = sets[set_name]['cost']

for node in G.nodes():
    constraint = {}
    for set_name, set_info in sets.items():
        if node in set_info['items']:
            constraint[set_name] = 1
    
    # Add quadratic terms for the constraint
    for set1 in constraint:
        for set2 in constraint:
            if set1 <= set2:
                Q[(set1, set2)] = Q.get((set1, set2), 0) - 20  # Penalty coefficient
    
    # Add linear terms for the constraint
    for set_name in constraint:
        Q[(set_name, set_name)] = Q.get((set_name, set_name), 0) + 10  # Penalty coefficient

# Convert QUBO to BinaryQuadraticModel
bqm = BinaryQuadraticModel.from_qubo(Q)

# Create a Neal sampler
sampler = Neal()

# Sample the QUBO
sampleset = sampler.sample(bqm, num_reads=1000)

# Get the best solution
best_solution = sampleset.first.sample

print(f"Graph: Nodes {G.nodes()}, Edges {G.edges()}")
print("\nSets:")
for set_name, set_info in sets.items():
    print(f"{set_name}: items {set_info['items']}, cost {set_info['cost']}")

print("\nOptimal Set Cover:")
selected_sets = [set_name for set_name, selected in best_solution.items() if selected == 1]
print(f"Selected sets: {selected_sets}")

# Verify coverage
covered_nodes = set()
for set_name in selected_sets:
    covered_nodes.update(sets[set_name]['items'])

print(f"Covered nodes: {covered_nodes}")
print(f"All nodes covered: {covered_nodes == set(G.nodes())}")
print(f"Total cost: {sum(sets[set_name]['cost'] for set_name in selected_sets)}")
