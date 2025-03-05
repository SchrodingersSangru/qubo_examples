import numpy as np
import random
from dwave_neal import Neal
from dimod import BinaryQuadraticModel

# Generate random number of assets (between 5 and 15)
num_assets = random.randint(5, 15)

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

# Convert QUBO to BinaryQuadraticModel
bqm = BinaryQuadraticModel.from_qubo(Q)

# Create a Neal sampler
sampler = Neal()

# Sample the QUBO
sampleset = sampler.sample(bqm, num_reads=1000)

# Get the best solution
best_solution = sampleset.first.sample

print(f"Number of assets: {num_assets}")
print(f"Optimal portfolio: {best_solution}")
print(f"Expected return: {sum(returns[i] for i in range(num_assets) if best_solution[i] == 1):.4f}")
print(f"Risk: {sum(cov_matrix[i][j] for i in range(num_assets) for j in range(num_assets) if best_solution[i] == 1 and best_solution[j] == 1):.6f}")
