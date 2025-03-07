import neal
import random
import numpy as np

# Generate random flights
num_flights = 30
flights = [f"F{i}" for i in range(num_flights)]

# Generate random crew members
num_crew = 15
crew = [f"C{i}" for i in range(num_crew)]

# Generate random routes
num_routes = 10
routes = [f"R{i}" for i in range(num_routes)]

# Generate random availability matrix
availability = np.random.randint(0, 2, size=(num_crew, num_flights))

# Create a binary quadratic model
bqm = {}

# Add linear terms (flight preferences for routes and crew preferences)
for i in range(num_routes):
    for j in range(num_flights):
        bqm[(i, j)] = -random.uniform(0, 1)  # Route-flight preference
    for k in range(num_crew):
        bqm[(i, k + num_flights)] = -random.uniform(0, 1) * availability[k][i % num_flights]  # Crew-route preference

# Add quadratic terms (constraints)
for i in range(num_routes):
    for j in range(num_flights):
        for k in range(j+1, num_flights):
            bqm[((i, j), (i, k))] = 2  # Penalty for assigning multiple flights to same route

for i in range(num_routes):
    for j in range(num_crew):
        for k in range(j+1, num_crew):
            bqm[((i, j + num_flights), (i, k + num_flights))] = 2  # Penalty for assigning multiple crew to same route

# Create a simulated annealing sampler
sampler = neal.SimulatedAnnealingSampler()

# Sample the problem
sampleset = sampler.sample_ising(h={}, J=bqm, num_reads=100)

# Get the best solution
best_solution = sampleset.first.sample

# Print the results
print("Optimal Route Scheduling:")
for i in range(num_routes):
    assigned_flights = [flights[j] for j in range(num_flights) if best_solution.get((i, j), 0) == 1]
    assigned_crew = [crew[j] for j in range(num_crew) if best_solution.get((i, j + num_flights), 0) == 1]
    print(f"{routes[i]}: Flights: {', '.join(assigned_flights)} | Crew: {', '.join(assigned_crew)}")
