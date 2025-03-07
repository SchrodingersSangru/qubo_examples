import networkx as nx
import random
import neal
import dimod
import matplotlib

def create_random_graph(n, p):
    G = nx.erdos_renyi_graph(n, p)
    return G

def graph_to_bqm(G):
    # Create a binary quadratic model (BQM) from the graph
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    
    # Add interactions between nodes
    for u, v in G.edges():
        bqm.add_interaction(u, v, 1)
    
    # Add a penalty term to encourage balanced partitions
    for node in G.nodes():
        bqm.add_variable(node, -1)
    
    return bqm

def partition_from_sample(sample):
    partition_0 = set(node for node, value in sample.items() if value == 0)
    partition_1 = set(node for node, value in sample.items() if value == 1)
    return partition_0, partition_1

def cut_size(G, partition):
    cut = 0
    for u, v in G.edges():
        if (u in partition[0] and v in partition[1]) or (u in partition[1] and v in partition[0]):
            cut += 1
    return cut

# Parameters
n = 100  # number of nodes
p = 0.1  # probability of edge creation
num_reads = 1000  # number of samples to collect

# Create random graph
G = create_random_graph(n, p)

nx.draw(G, with_labels=True)

# Convert graph to BQM
bqm = graph_to_bqm(G)

# Create a simulated annealing sampler
sampler = neal.SimulatedAnnealingSampler()

# Run simulated annealing
sampleset = sampler.sample(bqm, num_reads=num_reads)

# Get the best sample
best_sample = sampleset.first.sample

# Convert the best sample to a partition
best_partition = partition_from_sample(best_sample)

# Calculate the cut size of the best partition
best_cut = cut_size(G, best_partition)

print(f"Best partition cut size: {best_cut}")
print(f"Partition sizes: {len(best_partition[0])}, {len(best_partition[1])}")
