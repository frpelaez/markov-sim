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
    beta = 2.5
    gamma = 1.0

    x0 = 1
    T = 10.0
    rng = np.random.default_rng()

    q_means = np.zeros(shape=(N + 1,))
    q_means[0] = 0
    P = np.zeros(shape=(N + 1, N + 1))
    P[0][0] = 1.0
    for i in range(1, N):
        recovery = gamma * i
        infection = beta * i * (N - i) / N
        q_means[i] = 1 / (recovery + infection)
        P[i, i - 1] = recovery / (recovery + infection)
        P[i, i + 1] = infection / (recovery + infection)
    q_means[N] = 1 / (gamma * N)
    P[N][N - 1] = 1.0

    times, states = Simulation.CTMC_sim(P, q_means, x0, T, rng=rng)

    plt.step(times, states, where="pre")
    plt.xlabel("Tiempo t", fontweight="bold")
    plt.ylabel("Número de infectados I(t)", fontweight="bold")
    plt.title("Evolución del Modelo SIS", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.grid(True, alpha=0.5)
    plt.show()

    est_dist = Simulation.CTMC_estimate_disttribution(
        P, q_means, x0, T, trials=1000
    )  # tarda un rato para trials grande
    print(
        f"Probability of epidemic ({gamma=}, {beta=}) persisting after {T=}ut:",
        1 - est_dist[0],
    )


if __name__ == "__main__":
    main()
