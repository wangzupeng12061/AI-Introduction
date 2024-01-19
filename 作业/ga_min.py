import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 25
POP_SIZE = 250
CROSS_RATE = 0.6
MUTA_RATE = 0.02
Iterations = 50
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


def select_population(pop, fitness):
    idx = np.random.choice(
        np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness / fitness.sum()
    )
    return pop[idx]


def crossover_and_mutate(pop):
    new_pop = []
    for i in pop:
        temp = i.copy()
        if np.random.rand() < CROSS_RATE:
            j = pop[np.random.randint(POP_SIZE)]
            cpoint1, cpoint2 = np.sort(np.random.randint(0, DNA_SIZE, 2))
            temp[cpoint1:cpoint2] = j[cpoint1:cpoint2]
        mutate(temp)
        new_pop.append(temp)
    return np.array(new_pop)


def mutate(dna):
    if np.random.rand() < MUTA_RATE:
        mutate_point = np.random.randint(0, DNA_SIZE)
        dna[mutate_point] ^= 1


def display_results(pop):
    fitness = get_fitness(pop)
    best_dna = pop[np.argmin(fitness)]
    x_best = decode_dna(best_dna)
    print(f"min_fitness: {min(fitness)}")
    print(f"最优的基因型：{best_dna}")
    print(f"x: {x_best}")
    print(f"F(x)_min = {function_to_optimize(x_best)}")


def plot_curve():
    X = np.linspace(*X_BOUND, 200)
    Y = function_to_optimize(X)
    plt.plot(X, Y)
    plt.xlabel("x")
    plt.ylabel("y")


if __name__ == "__main__":
    plt.ion()
    plot_curve()

    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))
    for _ in range(Iterations):
        x = decode_dna(pop)
        if "sca" in locals():
            sca.remove()
        sca = plt.scatter(x, function_to_optimize(x), c="black", marker="o")
        plt.pause(0.1)
        pop = crossover_and_mutate(pop)
        fitness = get_fitness(pop)
        pop = select_population(pop, fitness)

    display_results(pop)
    plt.ioff()
    plt.show()
