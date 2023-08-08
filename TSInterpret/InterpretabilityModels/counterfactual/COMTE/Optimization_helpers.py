import numpy as np


def random_hill_climb(
    problem,
    max_attempts=10,
    max_iters=np.inf,
    restarts=0,
    init_state=None,
    curve=False,
    random_state=None,
):
    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    best_fitness = np.inf
    best_state = None

    if curve:
        fitness_values = []

    for _ in range(restarts + 1):
        # Initialize optimization problem and attempts counter
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)

        attempts = 0
        iters = 0

        while (attempts < max_attempts) and (iters < max_iters):
            iters += 1

            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            if next_fitness < problem.get_fitness():
                problem.set_state(next_state)
                attempts = 0

            else:
                attempts += 1

            if curve:
                fitness_values.append(problem.get_fitness())

        # Update best state and best fitness
        # print('best_fitness',best_fitness)
        if problem.get_fitness() < best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()
            # print('bestfitness after', best_fitness)

    if curve:
        import matplotlib.pyplot as plt

        plt.plot(np.asarray(fitness_values))
        plt.show()

    return best_state, best_fitness
