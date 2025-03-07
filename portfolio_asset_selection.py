import numpy as np
import random
from neal import Neal
from dimod import BinaryQuadraticModel

# Generate random number of assets (between 5 and 15)
num_assets = 11

# Generate random returns and covariance matrix
returns = np.random.uniform(0.05, 0.25, num_assets)
cov_matrix = np.random.uniform(0.001, 0.01, (num_assets, num_assets))
cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Make it symmetric
np.fill_diagonal(cov_matrix, np.random.uniform(0.003, 0.008, num_assets))  # Set diagonal values

# Formulate QUBO
Q = {}
for i in range(num_assets):
    for j in range(num_assets):
        if i <= j:
            Q[(i, j)] = cov_matrix[i][j]
    Q[(i, i)] -= 2 * returns[i]

# Add diversity penalty
diversity_penalty = max(abs(x) for x in Q.values()) * 0.1
for i in range(num_assets - 1):
    Q[(i, i+1)] = Q.get((i, i+1), 0) + diversity_penalty

# Convert QUBO to BinaryQuadraticModel
bqm = BinaryQuadraticModel.from_qubo(Q)

# Add constraint to select half of the assets
target_num_assets = num_assets // 2
lagrange_multiplier = max(abs(x) for x in Q.values()) * 10

for i in range(num_assets):
    bqm.add_variable(i, lagrange_multiplier)

bqm.add_linear_equality_constraint(
    [(i, 1) for i in range(num_assets)],
    constant=-target_num_assets,
    lagrange_multiplier=lagrange_multiplier
)

# Create a Neal sampler
sampler = Neal()

# Sample the QUBO
sampleset = sampler.sample(bqm, num_reads=1000)
# print(sampleset)

# Get the best solution
best_solution = sampleset.first.sample

best_solution_array = np.array([best_solution[i] for i in range(num_assets)])

print(f"Number of assets: {num_assets}")
# print(f"Optimal portfolio: {best_solution}")
print(f"Opimal assets selected : {best_solution_array}")

print(f"Selected assets: {sum(best_solution.values())}")
print(f"Expected return: {sum(returns[i] for i in range(num_assets) if best_solution[i] == 1):.4f}")
print(f"Risk: {sum(cov_matrix[i][j] for i in range(num_assets) for j in range(num_assets) if best_solution[i] == 1 and best_solution[j] == 1):.6f}")
