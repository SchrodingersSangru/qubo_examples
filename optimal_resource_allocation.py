import neal
import numpy as np
import dimod

# Generate random problem data
num_tasks = 5
num_resources = 3
num_time_slots = 4

# Random priorities for tasks (higher is more important)
task_priorities = np.random.randint(1, 10, size=num_tasks)

# Random costs for using resources
resource_costs = np.random.randint(1, 5, size=num_resources)

# Random production values for tasks
task_production = np.random.randint(5, 15, size=num_tasks)

print("\nTask Priorities:", task_priorities)
print("Resource Costs:", resource_costs)
print("Task Production Values:", task_production)


# Create QUBO matrix
Q = {}

# Objective: Maximize priority * production - cost
for t in range(num_tasks):
    for r in range(num_resources):
        for s in range(num_time_slots):
            idx = (t, r, s)
            Q[(idx, idx)] = -1 * (task_priorities[t] * task_production[t] - resource_costs[r])

# Constraint: Each task must be assigned exactly once
constraint_weight = 1000  # Adjust this weight to make constraints stronger
for t in range(num_tasks):
    task_sum = sum(Q.get(((t, r, s), (t, r, s)), 0) for r in range(num_resources) for s in range(num_time_slots))
    Q[((t, 0, 0), (t, 0, 0))] += constraint_weight * (1 - 2 * task_sum)
    for r1 in range(num_resources):
        for s1 in range(num_time_slots):
            for r2 in range(num_resources):
                for s2 in range(num_time_slots):
                    if (r1, s1) < (r2, s2):
                        Q[((t, r1, s1), (t, r2, s2))] = Q.get(((t, r1, s1), (t, r2, s2)), 0) + constraint_weight

# Constraint: Each resource can only be used once per time slot
for r in range(num_resources):
    for s in range(num_time_slots):
        resource_sum = sum(Q.get(((t, r, s), (t, r, s)), 0) for t in range(num_tasks))
        Q[((0, r, s), (0, r, s))] += constraint_weight * (1 - 2 * resource_sum)
        for t1 in range(num_tasks):
            for t2 in range(t1 + 1, num_tasks):
                Q[((t1, r, s), (t2, r, s))] = Q.get(((t1, r, s), (t2, r, s)), 0) + constraint_weight

# Create a BQM from the QUBO dictionary
bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

# Solve using SimulatedAnnealingSampler
sampler = neal.SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=1000, num_sweeps=1000)

# Get the best solution
best_solution = sampleset.first.sample

# Print the results
print("\n")

print("Best solution:")
assignments = []
total_priority = 0
total_cost = 0
total_production = 0

for (t, r, s), value in best_solution.items():
    if value == 1:
        assignments.append((t, r, s))
        total_priority += task_priorities[t]
        total_cost += resource_costs[r]
        total_production += task_production[t]

# Sort assignments by time slot for better readability
assignments.sort(key=lambda x: (x[2], x[1], x[0]))

for t, r, s in assignments:
    print(f"Task {t} (Priority: {task_priorities[t]}, Production: {task_production[t]}) assigned to Resource {r} (Cost: {resource_costs[r]}) at Time Slot {s}")

print(f"\nTotal Priority: {total_priority}")
print(f"Total Cost: {total_cost}")
print(f"Total Production: {total_production}")
print(f"Objective Value: {total_priority * total_production - total_cost}")


