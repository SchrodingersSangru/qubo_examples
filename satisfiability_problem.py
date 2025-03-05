import random
from dimod import BinaryQuadraticModel
from dwave_neal import Neal

# Function to generate a random 3-SAT problem
def generate_random_3sat(num_variables, num_clauses):
    clauses = []
    for _ in range(num_clauses):
        clause = []
        for _ in range(3):  # Each clause has 3 literals
            var = random.randint(1, num_variables)
            negated = random.choice([True, False])
            clause.append((-var if negated else var))
        clauses.append(clause)
    return clauses

# Convert 3-SAT to QUBO
def convert_3sat_to_qubo(clauses, penalty_weight=1):
    Q = {}
    for clause in clauses:
        x1, x2, x3 = clause

        # Adjust indices to be zero-based and handle negation
        x1_idx, x2_idx, x3_idx = abs(x1) - 1, abs(x2) - 1, abs(x3) - 1
        x1_sign, x2_sign, x3_sign = (x1 > 0), (x2 > 0), (x3 > 0)

        # Add penalty terms for unsatisfied clauses
        Q[(x1_idx, x1_idx)] = Q.get((x1_idx, x1_idx), 0) + penalty_weight * (not x1_sign)
        Q[(x2_idx, x2_idx)] = Q.get((x2_idx, x2_idx), 0) + penalty_weight * (not x2_sign)
        Q[(x3_idx, x3_idx)] = Q.get((x3_idx, x3_idx), 0) + penalty_weight * (not x3_sign)

        Q[(x1_idx, x2_idx)] = Q.get((x1_idx, x2_idx), 0) - penalty_weight
        Q[(x2_idx, x3_idx)] = Q.get((x2_idx, x3_idx), 0) - penalty_weight
        Q[(x1_idx, x3_idx)] = Q.get((x1_idx, x3_idx), 0) - penalty_weight

    return Q

# Solve the SAT problem using dwave-neal
def solve_sat_with_neal(Q):
    bqm = BinaryQuadraticModel.from_qubo(Q)
    sampler = Neal()
    sampleset = sampler.sample(bqm, num_reads=100)
    return sampleset.first.sample

# Main program
if __name__ == "__main__":
    num_variables = int(input("Enter the number of variables: "))
    num_clauses = int(input("Enter the number of clauses: "))

    # Generate random 3-SAT problem
    clauses = generate_random_3sat(num_variables, num_clauses)
    print("\nRandomly Generated 3-SAT Problem:")
    for clause in clauses:
        print(f"({clause[0]} OR {clause[1]} OR {clause[2]})")

    # Convert to QUBO and solve
    qubo = convert_3sat_to_qubo(clauses)
    solution = solve_sat_with_neal(qubo)

    # Print results
    print("\nSolution:")
    for var in range(num_variables):
        print(f"x{var + 1} =", solution[var])
