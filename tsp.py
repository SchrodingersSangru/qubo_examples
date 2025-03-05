import numpy as np
import random
from dwave_neal import Neal
from dimod import BinaryQuadraticModel

# Get number of cities from user
num_cities = int(input("Enter the number of cities: "))

# Generate random distance matrix
distances = np.random.randint(1, 100, size=(num_cities, num_cities))
np.fill_diagonal(distances, 0)  # Set diagonal to 0

# Ensure symmetry
distances = (distances + distances.T) // 2

print("Distance Matrix:")
print(distances)

# Formulate QUBO
Q = {}
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            # Linear terms: distances
            Q[(f'{i}_{j}', f'{i}_{j}')] = distances[i][j]

            # Quadratic terms: constraints
            for k in range(num_cities):
                if k != i and k != j:
                    Q[(f'{i}_{j}', f'{i}_{k}')] = 1000  # Each city visited once
                    Q[(f'{i}_{j}', f'{k}_{j}')] = 1000  # Each position filled once

# Convert QUBO to BinaryQuadraticModel
bqm = BinaryQuadraticModel.from_qubo(Q)

# Create a Neal sampler
sampler = Neal()

# Sample the QUBO
sampleset = sampler.sample(bqm, num_reads=1000)

# Get the best solution
best_solution = sampleset.first.sample

# Process the results
tour = [-1] * num_cities
for (i_j, value) in best_solution.items():
    if value == 1:
        i, j = map(int, i_j.split('_'))
        tour[j] = i

# Complete the tour
for i in range(num_cities):
    if i not in tour:
        tour[tour.index(-1)] = i

print("\nBest Tour Found:")
print(" -> ".join(map(str, tour + [tour[0]])))

# Calculate total distance
total_distance = sum(distances[tour[i]][tour[(i+1) % num_cities]] for i in range(num_cities))
print(f"\nTotal Distance: {total_distance}")
