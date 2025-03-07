import neal
import random
import numpy as np

# Generate random flights
num_flights = 20
flights = [f"F{i}" for i in range(num_flights)]

# Generate random crew members
num_crew = 10
crew = [f"C{i}" for i in range(num_crew)]

# Generate random availability matrix
availability = np.random.randint(0, 2, size=(num_crew, num_flights))

# Create a binary quadratic model
bqm = {}

# Add linear terms (crew preferences)
for i in range(num_crew):
    for j in range(num_flights):
        bqm[(i, j)] = -random.uniform(0, 1) * availability[i][j]

# Add quadratic terms (constraints)
for i in range(num_flights):
    for j in range(num_crew):
        for k in range(j+1, num_crew):
            bqm[((j, i), (k, i))] = 2  # Penalty for assigning multiple crew to same flight

for i in range(num_crew):
    for j in range(num_flights):
        for k in range(j+1, num_flights):
            bqm[((i, j), (i, k))] = 2  # Penalty for assigning same crew to consecutive flights

# Create a simulated annealing sampler
sampler = neal.SimulatedAnnealingSampler()

# Sample the problem
sampleset = sampler.sample_ising(h={}, J=bqm, num_reads=100)

# Get the best solution
best_solution = sampleset.first.sample

# Print the results
print("Optimal Crew Scheduling:")
for i in range(num_flights):
    assigned_crew = [crew[j] for j in range(num_crew) if best_solution.get((j, i), 0) == 1]
    print(f"{flights[i]}: {', '.join(assigned_crew)}")
