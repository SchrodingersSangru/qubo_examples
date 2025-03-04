import numpy as np
from qubovert import QUBO
from qubovert.sim import anneal_qubo

# Assume returns is a vector of expected returns
# and cov_matrix is the covariance matrix of returns
returns = np.array([0.1, 0.2, 0.15, 0.1])
cov_matrix = np.array([[0.005, 0.001, 0.002, 0.001],
                       [0.001, 0.006, 0.001, 0.002],
                       [0.002, 0.001, 0.004, 0.001],
                       [0.001, 0.002, 0.001, 0.003]])

# Formulate QUBO
Q = QUBO()
for i in range(len(returns)):
    for j in range(len(returns)):
        Q[(i, j)] = cov_matrix[i][j]
    Q[(i, i)] -= 2 * returns[i]

# Solve using simulated annealing
result = anneal_qubo(Q, num_anneals=10)
portfolio = result.best.state

print(portfolio)
