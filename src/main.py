import numpy as np

from simulation import Simulation


def main():
    trials = 25
    steps = 101
    n_states = 50
    a = np.zeros(n_states)
    a[15] = 1.0
    t = np.zeros((n_states, n_states))
    t[0][0] = 1
    for i in range(1, n_states - 1):
        t[i][i - 1] = 0.25
        t[i][i] = 0.25
        t[i][i + 1] = 0.5
    t[-1][-1] = 1

    sim = Simulation(trials, steps, n_states, a, t)
    sim.run(verbose=False, show_plots=True)
    # res = sim.result


if __name__ == "__main__":
    main()
