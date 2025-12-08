import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar

from errors import InvalidDistributionError, ParameterError
from utils import matpow

DISTRIBUTION_TOLERANCE = 1e-9


@dataclass
class SimulationResultWrapper:
    """
    ### Object that encapsulates the results of a DTMC simulation.
    """

    paths: list[list[int]]
    exact_final_dist: np.ndarray
    exact_final_mat: np.ndarray
    final_dist: np.ndarray | None = None
    final_mat: np.ndarray | None = None


class Simulation:
    """
    ### Class to encapsulate a DTMC simulation.
    The user must provide the following parameters:
    - numer of trials for the simulation (int)
    - number of steps to run each trial (int)
    - number of states (int)
    - the initial probability distribution (ArrayLike)
    - the one-epoch transition matrix (ArrayLike)

    The `run` method of this class will populate an atribute called `result`. This is an object containing important information and metrics about the experiment. The most important is `paths` (a list containing the generated trajectories).
    """

    def __init__(
        self,
        trials: int,
        steps: int,
        n_states: int,
        initial_distribution: np.ndarray,
        transition_matrix: np.ndarray,
        estimate_distribution: bool = False,
        seed: int = 12345,
    ) -> None:
        self.trials = trials
        self.steps = steps
        self.state_space = [i for i in range(n_states)]
        self.n_states = n_states
        self.initial_distribution = initial_distribution
        self.transition_matrix = transition_matrix
        self.result: SimulationResultWrapper = SimulationResultWrapper(
            [], np.array([]), np.array([])
        )
        self._est_dist = estimate_distribution
        self._rng = np.random.RandomState(seed=seed)

    def run(self, verbose: bool = False, show_plots: bool = False) -> None:
        """
        Runs the simulation given the initialization parameters. Can print to the screen the simulation parameters, but this is not reccomended for larger experiments. Can also generate automathic graphics to illustrate the results of the simulation.

        **Parameters**
        - verbose (bool, optional) Show the initialization parameters. Defaults to False
        - show_plots (bool, optional) Show the generated plots. Defaults to False
        """
        if not self._check_dimensions():
            msg = "Simulations parameters dimensions are incompatible."
            ste = f"State space has dimensions ({self.n_states},)"
            idt = f"initial distribution has dimensions ({len(self.initial_distribution)},) and"
            tmt = f"transitions matrix has dimensions {self.transition_matrix.shape}"
            raise ParameterError(" ".join([msg, ste, idt, tmt]))

        if not self._check_valid_distributions():
            raise InvalidDistributionError(
                "Either inital distribution or transition matrix are not valid."
                + " They must represent, row wise, probability distributions, aka, add up to 1"
            )

        if verbose:
            self._print_parameters()

        paths = self._compute_paths()

        final_dist = None
        final_mat = None
        if self._est_dist:
            final_mat = self._estimate_final_transition_matrix(trials=5000)
            final_dist = self.initial_distribution @ final_mat

        exact_final_mat = self._calculate_exact_final_matrix()
        exact_final_dist = self.initial_distribution @ exact_final_mat

        self.result = SimulationResultWrapper(
            paths, exact_final_dist, exact_final_mat, final_dist, final_mat
        )

        if show_plots:
            self._show_plots()

    @staticmethod
    def CTMC_sim(
        P: np.ndarray,
        q_means: np.ndarray,
        x0: int,
        T: float,
        a: np.ndarray | None = None,
        rng: np.random.RandomState | None = None,
    ) -> tuple[list[float], list[int]]:
        if rng is None:
            rng = np.random.default_rng()

        P = np.asarray(P, dtype=float)
        q_means = np.asarray(q_means, dtype=float)
        N = P.shape[0]

        if P.shape != (N, N):
            raise ParameterError(
                f"P must be a NxN square matrix, but got shape {P.shape}"
            )
        if np.any(q_means < 0):
            raise ParameterError("All exponential means must be positive")
        if not (0 <= x0 < N):
            raise ParameterError("Initial state x0 must be inside [0, N-1]")
        if T <= 0:
            raise ParameterError("Final time T must be positive")
        if np.any(np.abs(P.sum(axis=1) - 1.0) > DISTRIBUTION_TOLERANCE):
            raise InvalidDistributionError("All the rows of P must add up to 1")

        if a is not None:
            if len(a) != N:
                raise ParameterError("a must be a (N,) array")
            if np.any(np.abs(a.sum() - 1.0) > DISTRIBUTION_TOLERANCE):
                raise InvalidDistributionError("Elements of a must add up to 1")
            x0 = rng.choice(list(range(N)), p=a)
        times = [0.0]
        states = [int(x0)]
        t = 0.0
        current = int(x0)
        while t < T:
            new_t = (
                t + rng.exponential(scale=q_means[current])
                if q_means[current] != 0
                else T + 1
            )
            next = rng.choice(list(range(N)), p=P[current])
            if new_t >= T:
                times.append(T)
                states.append(current)
                break
            times.append(new_t)
            states.append(next)
            t = new_t
            current = int(next)
        return times, states

    @staticmethod
    def CTMC_state_at(t: float, times: list[float], states: list[int]) -> int:
        if not (times[0] <= t <= times[-1]):
            raise ParameterError("t outside of bounds")
        n = len(times)
        i = 0
        while i < n - 1 and not (times[i] <= t < times[i + 1]):
            i += 1
        return states[i]

    @staticmethod
    def CTMC_estimate_disttribution(
        P: np.ndarray,
        q_means: np.ndarray,
        x0: int,
        T: float,
        rng: np.random.RandomState | None = None,
        trials: int = 1_000,
    ) -> np.ndarray:
        print("Estimating distribution...")
        N = P.shape[0]
        counts = np.zeros(shape=(N,))
        with alive_bar(trials) as bar:
            for _ in range(trials):
                _, states = Simulation.CTMC_sim(P, q_means, x0, T, rng=rng)
                counts[states[-1]] += 1
                bar()
        print("Done!")
        return counts / counts.sum()

    def estimate_distribution_from_initial_state(
        self,
        initial_state: int,
        epochs: int,
        n_trials: int = 1_000,
    ) -> np.ndarray:
        """
        Estimates the distribution after `epochs` epochs starting from `initial_state` by running the simulation `n_trials` times.
        """
        counts = np.zeros((self.n_states,))
        for _ in range(n_trials):
            state = initial_state
            for _ in range(1, epochs):
                state = self._rng.choice(
                    self.state_space, p=self.transition_matrix[state]
                )
            counts[state] += 1
        return counts / n_trials

    def _print_parameters(self) -> None:
        print(
            """
            --- Markov Chain Simulation ---
            """
        )
        print("    - Parameters")
        print(f"              Trials: {self.trials}")
        print(f"               Steps: {self.steps}")
        print(f"    State space size: {self.n_states}")
        print(f"Initial distribution: {self.initial_distribution}")
        print("   Transition matrix:")
        print(self.transition_matrix)

    def _show_plots(self) -> None:
        plt.figure(figsize=(12, 6))
        epochs = [i for i in range(0, self.steps)]
        if self.result is not None:
            for path in self.result.paths:
                plt.plot(epochs, path)

            plt.xlabel("Step")
            plt.ylabel("State")
            xtick_step = 1 if self.steps <= 20 else self.steps // 10
            ytick_step = 1 if self.n_states <= 20 else self.n_states // 10
            plt.xticks(list(range(0, self.steps, xtick_step)))
            plt.yticks(list(range(0, self.n_states, ytick_step)))
            plt.ylim(bottom=-1, top=len(self.state_space))
            plt.title("Markov Chain Simulation")
            plt.tight_layout()
            plt.show()

    def _compute_paths(self) -> list[list[int]]:
        paths = [[] for _ in range(self.trials)]
        for i in range(self.trials):
            path = [0 for _ in range(self.steps)]
            in_state = self._rng.choice(self.state_space, p=self.initial_distribution)
            path[0] = in_state
            pv_state = in_state
            for j in range(1, self.steps):
                cr_state = self._rng.choice(
                    self.state_space, p=self.transition_matrix[pv_state]
                )
                path[j] = cr_state
                pv_state = cr_state
            paths[i] = path
        return paths

    def _calculate_exact_final_matrix(self) -> np.ndarray:
        return matpow(self.transition_matrix, self.steps)

    def _estimate_final_transition_matrix(self, trials: int = 1_000) -> np.ndarray:
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    Simulation._simulation_worker,
                    range(self.n_states),
                    itertools.repeat(self.steps),
                    itertools.repeat(self.n_states),
                    itertools.repeat(trials),
                    itertools.repeat(self.transition_matrix),
                    itertools.repeat(12345),
                )
            )

        return np.stack(results)

    @staticmethod
    def _simulation_worker(
        initial_state: int,
        steps: int,
        n_states: int,
        trials: int,
        transition_matrix: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        counts = np.zeros((n_states,))
        rng = np.random.RandomState(seed=initial_state + seed)
        for _ in range(trials):
            state = initial_state
            for _ in range(1, steps):
                state = rng.choice(list(range(n_states)), p=transition_matrix[state])
            counts[state] += 1
        return counts / trials

    def _check_dimensions(self) -> bool:
        return self.transition_matrix.shape == (
            self.n_states,
            self.n_states,
        ) and self.n_states == len(self.initial_distribution)

    def _check_valid_distributions(self) -> bool:
        if np.abs(self.initial_distribution.sum() - 1.0) >= DISTRIBUTION_TOLERANCE:
            return False

        for i in range(self.transition_matrix.shape[0]):
            if np.abs(self.transition_matrix[i].sum() - 1.0) >= DISTRIBUTION_TOLERANCE:
                return False

        return True
