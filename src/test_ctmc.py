import matplotlib.pyplot as plt
import numpy as np

from simulation import Simulation


def main() -> None:
    P = np.random.random(size=(6, 6))
    P /= P.sum(axis=1)[:, np.newaxis]
    q_means = np.random.random(size=(6,))
    x0 = 0
    T = 20.0
    rng = None
    print(
        """
        --- CTMC Simulation ---
    """
    )
    print(f"{P=}")
    print(f"{q_means=}")
    times, states = Simulation.CTMC_sim(P, q_means, x0, T, rng)
    t = 7.0
    print(f"State at time {t=}: {Simulation.CTMC_state_at(t, times, states)}")
    trials = 1_000
    est_dist = Simulation.CTMC_estimate_disttribution(P, q_means, x0, T, rng, trials)
    print(f"Estimated distribution after {T=}s: {est_dist}")
    plt.step(times, states, where="post")
    plt.title("CTMC simulation")
    plt.yticks(list(range(P.shape[0])))
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.grid(True, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
