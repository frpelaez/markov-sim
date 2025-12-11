import matplotlib.pyplot as plt
import numpy as np

from simulation import Simulation


def maquina() -> None:
    print(
        """
        --- Máquina Rota-Funcionando---
    """
    )
    N = 1
    lmd = 2.0
    mu = 1.0

    x0 = 1
    T = 20.0
    rng = np.random.default_rng()
    q_means = np.array([1 / lmd, 1 / mu])
    P = np.array([[0.0, 1.0], [1.0, 0.0]])

    times, states = Simulation.CTMC_sim(P, q_means, x0, T, rng=rng)
    est_dist = Simulation.CTMC_estimate_disttribution(P, q_means, x0, T, trials=100)
    print(f"Modelo de máquina Rota-Funcionando con tasas {lmd=}, {mu=}")

    plot_results(
        times,
        states,
        est_dist,
        T,
        N,
        "Evolución de la CMTC",
        "X(t)",
        width=0.6,
        xticks=[0, 1],
        xtickslabels=["Rota", "Funcionando"],
    )


def sis() -> None:
    print(
        """
        --- Modelo de epidemias SIS ---
    """
    )
    N = 100
    beta = 3.0
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
    est_dist = Simulation.CTMC_estimate_disttribution(P, q_means, x0, T, trials=100)
    print(
        f"Probability of epidemic ({gamma=}, {beta=}) persisting after {T=}:",
        1 - est_dist[0],
    )

    plot_results(
        times,
        states,
        est_dist,
        T,
        N,
        "Evolución del modelo SIS",
        "Número de infectados I(t)",
    )


def plot_results(
    times,
    states,
    dist,
    T,
    N,
    title,
    ylabel,
    xlabel="Tiempo t",
    width=1.5,
    xticks=None,
    xtickslabels=None,
):
    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].step(times, states, where="pre")
    axes[0].set_xlabel(xlabel or "Tiempo t", fontweight="bold")
    axes[0].set_ylabel(ylabel or "Número de infectados I(t)", fontweight="bold")
    axes[0].set_title(
        title or "Evolución del Modelo SIS", fontsize=12, fontweight="bold"
    )
    axes[0].grid(True, alpha=0.5)

    axes[1].bar(list(range(0, N + 1)), dist, width=width)
    if xticks is not None and xtickslabels is not None:
        axes[1].set_xticks(xticks, xtickslabels)
    axes[1].set_xlabel("Estados", fontweight="bold")
    axes[1].set_ylabel("Probabilidad", fontweight="bold")
    axes[1].set_title(
        f"Distribución de la CTMC a tiempo {T=}", fontsize=12, fontweight="bold"
    )
    axes[1].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()
