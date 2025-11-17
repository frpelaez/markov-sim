from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


class SimulationParameterError(Exception): ...


class SimulationInvalidDistributionError(Exception): ...


DISTRIBUTION_TOLERANCE = 1e-9


@dataclass
class SimulationResultWrapper:
    paths: list[list[int]]
    final_matrix: np.ndarray
    final_distribution: np.ndarray
    # TODO: añadir más métricas que permitan analizar la cadena


class Simulation:
    def __init__(
        self,
        trials: int,
        steps: int,
        n_states: int,
        initial_distribution: np.ndarray,
        transition_matrix: np.ndarray,
    ) -> None:
        self.trials = trials
        self.steps = steps
        self.state_space = [i for i in range(n_states)]
        self.n_states = n_states
        self.initial_distribution = initial_distribution
        self.transition_matrix = transition_matrix
        self.result: None | SimulationResultWrapper = None

    def run(self, verbose: bool = False, show_plots: bool = False) -> None:
        if not self._check_dimensions():
            msg = "Simulations parameters dimensions are incompatible."
            ste = f"State space has dimensions ({self.n_states},)"
            idt = f"initial distribution has dimensions ({len(self.initial_distribution)},) and"
            tmt = f"transitions matrix has dimensions {self.transition_matrix.shape}"
            raise SimulationParameterError(" ".join([msg, ste, idt, tmt]))

        if not self._check_valid_distributions():
            raise SimulationInvalidDistributionError(
                "Either inital distribution or transition matrix are not valid."
                + " They must represent, row ways, probability distributions, aka add up to 1"
            )

        if verbose:
            self._print_parameters()

        paths = self._compute_paths()
        final_mat, final_dist = self._calculate_distribution()
        self.result = SimulationResultWrapper(paths, final_mat, final_dist)

        if show_plots:
            self._show_plots()

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
        _ = plt.figure(figsize=(12, 6))
        epochs = [i for i in range(0, self.steps)]
        if self.result is not None:
            for path in self.result.paths:
                plt.plot(epochs, path)

            plt.xlabel("Step")
            plt.ylabel("State")
            plt.xticks(epochs)
            plt.yticks(self.state_space)
            plt.ylim(bottom=-1, top=len(self.state_space))
            plt.title("Markov Chain Simulation")
            plt.tight_layout()
            plt.show()

    def _compute_paths(self) -> list[list[int]]:
        paths = []
        for _ in range(self.trials):
            path = []
            in_state = np.random.choice(self.state_space, p=self.initial_distribution)
            path.append(in_state)
            pv_state = in_state
            for _ in range(1, self.steps):
                cr_state = np.random.choice(
                    self.state_space, p=self.transition_matrix[pv_state]
                )
                path.append(cr_state)
                pv_state = cr_state
            paths.append(path)
        return paths

    def _calculate_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        mat = self.transition_matrix.copy()
        # TODO: choricero que flipas xd
        for _ in range(self.steps):
            mat = np.matmul(mat, self.transition_matrix)
        for i in range(mat.shape[0]):
            mat[i] /= mat[i].sum()
        final_vec = np.dot(mat, self.initial_distribution)
        final_vec /= final_vec.sum()
        return mat, final_vec

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
