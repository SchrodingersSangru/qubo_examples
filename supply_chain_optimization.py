import random
import numpy as np
from dwave_neal import Neal
from dimod import BinaryQuadraticModel

# Generate random supply chain problem
num_suppliers = random.randint(3, 6)
num_products = random.randint(4, 8)

# Random production costs for each supplier-product combination
production_costs = np.random.uniform(10, 100, (num_suppliers, num_products))

# Random demand for each product
demand = np.random.randint(50, 200, num_products)

# Random capacity for each supplier
capacity = np.random.randint(100, 300, num_suppliers)

# Formulate QUBO
Q = {}
for i in range(num_suppliers):
    for j in range(num_products):
        # Linear terms: production costs
        Q[(f's{i}p{j}', f's{i}p{j}')] = production_costs[i][j]

        # Quadratic terms: demand constraints
        for k in range(num_suppliers):
            if k != i:
                Q[(f's{i}p{j}', f's{k}p{j}')] = 1000  # Penalty for overproduction

        # Quadratic terms: capacity constraints
        for l in range(num_products):
            if l != j:
                Q[(f's{i}p{j}', f's{i}p{l}')] = 10  # Soft penalty for exceeding capacity

# Convert QUBO to BinaryQuadraticModel
bqm = BinaryQuadraticModel.from_qubo(Q)

# Create a Neal sampler
sampler = Neal()

# Sample the QUBO
sampleset = sampler.sample(bqm, num_reads=1000)

# Get the best solution
best_solution = sampleset.first.sample

# Process the results
total_cost = 0
production_plan = np.zeros((num_suppliers, num_products), dtype=int)
for (supplier, product), value in best_solution.items():
    if value == 1:
        i = int(supplier[1:])
        j = int(product[1:])
        production_plan[i][j] = 1
        total_cost += production_costs[i][j]

print("Supply Chain Optimization Results:")
print(f"Number of suppliers: {num_suppliers}")
print(f"Number of products: {num_products}")
print("\nProduction Plan:")
print(production_plan)
print(f"\nTotal Production Cost: {total_cost:.2f}")

# Check constraints
print("\nConstraint Satisfaction:")
for j in range(num_products):
    produced = sum(production_plan[:, j])
    print(f"Product {j}: Demand = {demand[j]}, Produced = {produced}")

for i in range(num_suppliers):
    used_capacity = sum(production_plan[i, :])
    print(f"Supplier {i}: Capacity = {capacity[i]}, Used = {used_capacity}")
