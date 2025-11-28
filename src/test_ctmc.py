import matplotlib.pyplot as plt
import numpy as np

from simulation import Simulation


def main() -> None:
    print(
        """
        --- SIS Model Simulation ---
    """
    )
    N = 100
    beta = 3.0
    gamma = 1.0

    x0 = 1
    T = 50.0
    rng = np.random.default_rng()

    q_means = np.zeros(shape=(N,))
    q_means[0] = 1e-6
    for i in range(1, N):
        q_means[i] = 1 / (gamma * i) + N / (beta * i * (N - i))

    P = np.zeros(shape=(N, N))
    P[0][0] = 1.0
    for i in range(1, N - 1):
        P[i][i - 1] = gamma * i
        P[i][i + 1] = beta * i * (N - i) / N
    P[N - 1][N - 2] = gamma * (N - 1)
    P /= P.sum(axis=1)[:, np.newaxis]

    # print(P)
    # print(q_means)

    times, states = Simulation.CTMC_sim(
        P, q_means, x0, T, check_extintion=True, rng=rng
    )

    plt.step(times, states, where="pre")
    plt.show()

    est_dist = Simulation.CTMC_estimate_disttribution(P, q_means, x0, T, trials=1_000)
    print(est_dist)


if __name__ == "__main__":
    main()
