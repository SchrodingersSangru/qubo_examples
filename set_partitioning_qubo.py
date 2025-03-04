from qubovert import QUBO
from qubovert.sim import anneal_qubo

# Define the sets and their costs
sets = {
    'A': {'items': [1, 2, 3], 'cost': 5},
    'B': {'items': [2, 3, 4], 'cost': 4},
    'C': {'items': [3, 4, 5], 'cost': 3}
}

# Formulate QUBO
Q = QUBO()
for set_name in sets:
    Q[set_name] = sets[set_name]['cost']

for item in range(1, 6):
    constraint = QUBO()
    for set_name, set_info in sets.items():
        if item in set_info['items']:
            constraint[set_name] = 1
    Q += (constraint - 1)**2 * 10  # Penalty term

# Solve using simulated annealing
result = anneal_qubo(Q, num_anneals=10)
partition = result.best.state

print(result)
print("\n")
print(partition)
