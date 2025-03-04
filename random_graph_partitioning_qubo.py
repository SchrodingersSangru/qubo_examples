import networkx as nx
import random
import math

def create_random_graph(n, p):
    return nx.erdos_renyi_graph(n, p)

def initial_partition(G):
    nodes = list(G.nodes())
    random.shuffle(nodes)
    mid = len(nodes) // 2
    return set(nodes[:mid]), set(nodes[mid:])

def cut_size(G, partition):
    cut = 0
    for u, v in G.edges():
        if (u in partition[0] and v in partition[1]) or (u in partition[1] and v in partition[0]):
            cut += 1
    return cut

def simulated_annealing(G, initial_temp, cooling_rate, num_iterations):
    current_partition = initial_partition(G)
    current_cut = cut_size(G, current_partition)
    best_partition = current_partition
    best_cut = current_cut
    temp = initial_temp

    for _ in range(num_iterations):
        # Choose a random node to move
        part = random.choice([0, 1])
        if len(current_partition[part]) > 1:
            node = random.choice(list(current_partition[part]))
            new_partition = (
                current_partition[0] - {node} if part == 0 else current_partition[0] | {node},
                current_partition[1] | {node} if part == 0 else current_partition[1] - {node}
            )
            new_cut = cut_size(G, new_partition)

            # Decide whether to accept the new partition
            delta = new_cut - current_cut
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_partition = new_partition
                current_cut = new_cut
                if current_cut < best_cut:
                    best_partition = current_partition
                    best_cut = current_cut

        # Cool down the temperature
        temp *= cooling_rate

    return best_partition, best_cut

# Parameters
n = 100  # number of nodes
p = 0.1  # probability of edge creation
initial_temp = 10.0
cooling_rate = 0.995
num_iterations = 10000

# Create random graph
G = create_random_graph(n, p)

# Perform simulated annealing
best_partition, best_cut = simulated_annealing(G, initial_temp, cooling_rate, num_iterations)

print(f"Best partition cut size: {best_cut}")
print(f"Partition sizes: {len(best_partition[0])}, {len(best_partition[1])}")
