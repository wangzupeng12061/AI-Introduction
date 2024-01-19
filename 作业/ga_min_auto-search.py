import numpy as np
import matplotlib.pyplot as plt

# Parameters for the target GA
DNA_SIZE = 25
X_BOUND = [0, 10]


def function_to_optimize(x):
    return 10 * np.sin(5 * x) + 7 * abs(x - 5) + 10


def decode_dna(pop):
    return (
        pop.dot(2 ** np.arange(DNA_SIZE)[::-1])
        / float(2**DNA_SIZE - 1)
        * (X_BOUND[1] - X_BOUND[0])
        + X_BOUND[0]
    )


def get_fitness(pop):
    temp = function_to_optimize(decode_dna(pop))
    return -(temp - np.max(temp)) + 0.0001


def select_population(pop, fitness, pop_size):
    idx = np.random.choice(
        np.arange(pop_size), size=pop_size, replace=True, p=fitness / fitness.sum()
    )
    return pop[idx]


def mutate(dna, muta_rate):  # Add muta_rate as an argument
    if np.random.rand() < muta_rate:  # Use the passed muta_rate
        mutate_point = np.random.randint(0, len(dna))
        dna[mutate_point] ^= 1


def crossover_and_mutate(
    pop, cross_rate, pop_size, muta_rate
):  # Add muta_rate as an argument
    new_pop = []
    for i in pop:
        temp = i.copy()
        if np.random.rand() < cross_rate:
            j = pop[np.random.randint(pop_size)]
            cpoint1, cpoint2 = np.sort(np.random.randint(0, DNA_SIZE, 2))
            temp[cpoint1:cpoint2] = j[cpoint1:cpoint2]
        mutate(temp, muta_rate)  # Pass the muta_rate when calling mutate
        new_pop.append(temp)
    return np.array(new_pop)


def run_target_ga(pop_size, cross_rate, muta_rate, iterations):
    pop = np.random.randint(2, size=(pop_size, DNA_SIZE))
    for _ in range(iterations):
        pop = crossover_and_mutate(
            pop, cross_rate, pop_size, muta_rate
        )  # Pass the muta_rate when calling crossover_and_mutate
        fitness = get_fitness(pop)
        pop = select_population(pop, fitness, pop_size)
    best_dna = pop[np.argmin(fitness)]
    x_best = decode_dna(best_dna)
    best_result = function_to_optimize(x_best)
    return best_result


# Parameters for the meta-GA
META_DNA_SIZE = 16
META_POP_SIZE = 20
META_CROSS_RATE = 0.7
META_MUTA_RATE = 0.02
META_ITERATIONS = 10

PARAM_RANGES = {
    "POP_SIZE": (100, 300),
    "CROSS_RATE": (0.4, 0.9),
    "MUTA_RATE": (0.01, 0.05),
    "Iterations": (20, 100),
}


def decode_meta_dna(dna):
    splits = [4, 6, 8, 14]  # Bit splits for each parameter
    params = {}
    bits = np.split(dna, splits)

    params["POP_SIZE"] = int(
        PARAM_RANGES["POP_SIZE"][0]
        + bits[0].dot(2 ** np.arange(bits[0].size)[::-1])
        / (2**4 - 1)
        * (PARAM_RANGES["POP_SIZE"][1] - PARAM_RANGES["POP_SIZE"][0])
    )
    params["CROSS_RATE"] = PARAM_RANGES["CROSS_RATE"][0] + bits[1].dot(
        2 ** np.arange(bits[1].size)[::-1]
    ) / (2**2 - 1) * (PARAM_RANGES["CROSS_RATE"][1] - PARAM_RANGES["CROSS_RATE"][0])
    params["MUTA_RATE"] = PARAM_RANGES["MUTA_RATE"][0] + bits[2].dot(
        2 ** np.arange(bits[2].size)[::-1]
    ) / (2**2 - 1) * (PARAM_RANGES["MUTA_RATE"][1] - PARAM_RANGES["MUTA_RATE"][0])
    params["Iterations"] = int(
        PARAM_RANGES["Iterations"][0]
        + bits[3].dot(2 ** np.arange(bits[3].size)[::-1])
        / (2**6 - 1)
        * (PARAM_RANGES["Iterations"][1] - PARAM_RANGES["Iterations"][0])
    )

    return params


def meta_fitness(dna):
    params = decode_meta_dna(dna)
    best_result = run_target_ga(
        params["POP_SIZE"],
        params["CROSS_RATE"],
        params["MUTA_RATE"],
        params["Iterations"],
    )
    return -best_result


best_fitness_values = []  # To store the best fitness value of each generation

if __name__ == "__main__":
    meta_pop = np.random.randint(2, size=(META_POP_SIZE, META_DNA_SIZE))
    meta_pop = crossover_and_mutate(
        meta_pop, META_CROSS_RATE, META_POP_SIZE, META_MUTA_RATE
    )

    best_params = None
    best_score = float("inf")

    for _ in range(META_ITERATIONS):
        fitness_values = np.array([meta_fitness(dna) for dna in meta_pop])
        current_best_fitness = min(fitness_values)
        best_fitness_values.append(current_best_fitness)  # Save the best fitness value of this generation
        if current_best_fitness < best_score:
            best_score = current_best_fitness
            best_params = decode_meta_dna(meta_pop[np.argmin(fitness_values)])
        meta_pop = select_population(meta_pop, -fitness_values, META_POP_SIZE)
        meta_pop = crossover_and_mutate(
            meta_pop, META_CROSS_RATE, META_POP_SIZE, META_MUTA_RATE
        )

    print("Best Parameters Found by Meta GA:")
    print(best_params)
    print("Best Score (negative of target function's result):", best_score)

    # Visualization of the search process
    plt.plot(best_fitness_values)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Meta-GA Optimization Process')
    plt.show()
