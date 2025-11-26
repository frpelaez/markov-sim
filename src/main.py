import numpy as np

from simulation import Simulation


def main():
    trials = 31
    steps = 701
    n_states = 121

    starting_range = 5
    a = np.zeros((n_states))
    a[:starting_range] = np.random.random(starting_range)
    a[:starting_range] /= a[:starting_range].sum()

    t = np.zeros((n_states, n_states))
    t[0][1] = 1
    for i in range(1, n_states - 1):
        t[i][i - 1] = 0.23
        t[i][i] = 0.43
        t[i][i + 1] = 0.34
    t[-1][-2] = 1

    estimate = False
    show_results = False

    sim = Simulation(
        trials, steps, n_states, a, t, estimate_distribution=estimate, seed=1
    )
    sim.run(verbose=False, show_plots=True)
    res = sim.result

    if estimate and show_results:
        print(
            """
              Estimations
              """
        )
        print("Final mat:\n", res.final_mat)
        print("Final dist:\n", res.final_dist)

    if show_results:
        print(
            """
              Exacts
              """
        )
        print("Exact mat:\n", res.exact_final_mat)
        print("Exact dist:\n", res.exact_final_dist)

    if res.final_mat is not None and res.final_dist is not None:
        mat_error = np.abs(res.exact_final_mat - res.final_mat).mean()
        dist_error = np.abs(res.exact_final_dist - res.final_dist).mean()
        print("      Mean absolute error in matrix:", mat_error)
        print("Mean absolute error in distribution:", dist_error)


if __name__ == "__main__":
    main()
