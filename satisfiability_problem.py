import random
import numpy as np
from dimod import BinaryQuadraticModel
from neal import Neal

def generate_random_3sat(num_variables, num_clauses):
    clauses = []
    for _ in range(num_clauses):
        clause = random.sample(range(1, num_variables + 1), 3)
        clause = [x if random.random() < 0.5 else -x for x in clause]
        clauses.append(clause)
    return clauses

def convert_3sat_to_qubo(clauses, penalty_weight=2):
    Q = {}
    for clause in clauses:
        x1, x2, x3 = clause
        x1_idx, x2_idx, x3_idx = abs(x1) - 1, abs(x2) - 1, abs(x3) - 1
        x1_sign, x2_sign, x3_sign = (x1 > 0), (x2 > 0), (x3 > 0)

        Q[(x1_idx, x1_idx)] = Q.get((x1_idx, x1_idx), 0) + penalty_weight * (not x1_sign)
        Q[(x2_idx, x2_idx)] = Q.get((x2_idx, x2_idx), 0) + penalty_weight * (not x2_sign)
        Q[(x3_idx, x3_idx)] = Q.get((x3_idx, x3_idx), 0) + penalty_weight * (not x3_sign)

        Q[(x1_idx, x2_idx)] = Q.get((x1_idx, x2_idx), 0) - penalty_weight * (x1_sign == x2_sign)
        Q[(x2_idx, x3_idx)] = Q.get((x2_idx, x3_idx), 0) - penalty_weight * (x2_sign == x3_sign)
        Q[(x1_idx, x3_idx)] = Q.get((x1_idx, x3_idx), 0) - penalty_weight * (x1_sign == x3_sign)

    # Add balance constraint
    for i in range(num_variables):
        for j in range(i+1, num_variables):
            Q[(i, j)] = Q.get((i, j), 0) + 0.1  # Small penalty for imbalance

    return Q

def solve_sat_with_neal(Q):
    bqm = BinaryQuadraticModel.from_qubo(Q)
    sampler = Neal()
    sampleset = sampler.sample(bqm, num_reads=1000)
    return sampleset.first.sample

if __name__ == "__main__":
    num_variables = 20
    num_clauses = 80  # Increased number of clauses

    clauses = generate_random_3sat(num_variables, num_clauses)
    print("\nRandomly Generated 3-SAT Problem:")
    for clause in clauses[:10]:  # Print first 10 clauses
        print(f"({clause[0]} OR {clause[1]} OR {clause[2]})")
    print("...")

    qubo = convert_3sat_to_qubo(clauses)
    solution = solve_sat_with_neal(qubo)

    print("\nSolution:")
    

    # Verify solution
    satisfied_clauses = sum(any((x > 0) == solution[abs(x)-1] for x in clause) for clause in clauses)
    print(f"\nSatisfied clauses: {satisfied_clauses} out of {num_clauses}")
    print(f"Satisfaction rate: {satisfied_clauses/num_clauses:.2%}")
