import numpy as np

from simulation import Simulation
from utils import matpow


def main():
    trials = 3
    steps = 101
    n_states = 5

    a = np.zeros((n_states,))
    a[0] = 1.0

    t = np.zeros((n_states, n_states))
    t[0][1] = 1
    for i in range(1, n_states - 1):
        t[i][i - 1] = 0.33
        t[i][i] = 0.33
        t[i][i + 1] = 0.34
    t[-1][-2] = 1

    sim = Simulation(trials, steps, n_states, a, t, estimate_distribution=True)
    sim.run(verbose=False, show_plots=False)
    res = sim.result

    print("""
          Estimations
          """)
    print("Final mat:\n", res.final_mat)
    print("Final dist:\n", res.final_dist)
    for i in range(n_states):
        dist = sim.estimate_distribution_from_initial_state(i, steps)
        print(f"Final dist from state {i}:\n", dist)

    print("""
          Exacts
          """)
    exact_mat = matpow(t, steps)
    for i in range(exact_mat.shape[0]):
        exact_mat[i] /= exact_mat[i].sum()
    exact_dist = a @ exact_mat
    print("Exact mat:\n", exact_mat)
    print("Exact dist:\n", exact_dist)


if __name__ == "__main__":
    main()
