import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
import neal
import random 


# Generate random values for the Knapsack problem
num_items = 10  # Number of items
values = [random.randint(10, 100) for _ in range(num_items)]  # Random values between 10 and 100
weights = [random.randint(5, 50) for _ in range(num_items)]  # Random weights between 5 and 50
capacity = random.randint(50, 200)  # Random capacity between 50 and 200

# Print the randomly generated values and weights
print("Values of items:", values)
print("Weights of items:", weights)
print("Knapsack capacity:", capacity)

# Number of items
n = len(values)

# Create the QUBO matrix (initialize with zeros)
Q = {}

# Add the quadratic terms (diagonal, for each item selected)
for i in range(n):
    Q[(i, i)] = -2 * values[i]  # Negative sign since we're maximizing the value

# Add the linear terms (penalty for exceeding capacity)
for i in range(n):
    for j in range(i + 1, n):
        Q[(i, j)] = 2 * (weights[i] + weights[j])  # Penalty for weight exceeding capacity

# Add the constraint for weight limit
weight_limit = capacity

# Adjust the Q matrix with penalties based on weight constraint
for i in range(n):
    for j in range(i + 1, n):
        if sum(weights[k] for k in range(n) if k == i or k == j) > weight_limit:
            Q[(i, j)] = 10  # High penalty if total weight exceeds limit

# Create a sampler using simulated annealing
sampler = neal.SimulatedAnnealingSampler()

# Sample solutions
sampleset = sampler.sample_qubo(Q, num_reads=10)

# Get the best solution
best_sample = sampleset.first.sample
print("Best solution:", best_sample)

# Calculate total value and weight for the best solution
selected_items = [i for i in range(n) if best_sample[i] == 1]
total_value = sum(values[i] for i in selected_items)
total_weight = sum(weights[i] for i in selected_items)

print(f"Selected items: {selected_items}")
print(f"Total value: {total_value}")
print(f"Total weight: {total_weight}")
