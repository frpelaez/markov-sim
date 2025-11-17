from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


class ParameterError(Exception): ...


class InvalidDistributionError(Exception): ...


DISTRIBUTION_TOLERANCE = 1e-9


@dataclass
class SimulationResultWrapper:
    paths: list[list[int]]
    estimated_final_distribution: np.ndarray
    # TODO: añadir más métricas que permitan analizar la cadena


class Simulation:
    """
    Class to encapsulate a DTMC simulation. The user must provide the following parameters:
    - numer of trials for the simulation (int)
    - number of steps to run each trial (int)
    - number of states (int)
    - the initial probability distribution
    - the one-epoch transition matrix

    The `run` method of this class will populate an atribute called `result`. This is an object containing important information and metrics about the experiment. Some of the most important are `paths` (a lsit containing the generated trajectories) and `final_distribution` (probability distribution over the states given the transition matrix and the initial distribution).
    """

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
        """
        Runs the simulation given the initialization parameters. Can print to the screen the simulation parameters, but this is not reccomended for larger experiments. Can also generate automathic graphics to illustrate the results of the simulation.

        :Parameters:
        -verbose (bool, optional) Show the initialization parameters. Defaults to False
        -show_plots (bool, optional) Show the generated plots. Defaults to False
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
                + " They must represent, row ways, probability distributions, aka add up to 1"
            )

        if verbose:
            self._print_parameters()

        paths = self._compute_paths()
        est_final_dist = self._estimate_final_distribution()
        self.result = SimulationResultWrapper(paths, est_final_dist)

        if show_plots:
            self._show_plots()

    def estimate_distribution_from_initial_state(
        self,
        initial_state: int,
        final_epoch: int,
        trials: int = 1_000,
    ) -> np.ndarray:
        """
        Estimates the distribution after `final_epochs` epochs starting from `initial_state` by running the simulation `trials` times
        """
        counts = np.zeros((self.n_states,))
        for _ in range(trials):
            state = initial_state
            for _ in range(1, final_epoch):
                state = np.random.choice(
                    self.state_space, p=self.transition_matrix[state]
                )
            counts[state] += 1
        return counts / counts.sum()

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
            plt.xticks(list(range(0, self.steps, self.steps // 10)))
            plt.yticks(list(range(0, self.n_states, self.n_states // 10)))
            plt.ylim(bottom=-1, top=len(self.state_space))
            plt.title("Markov Chain Simulation")
            plt.tight_layout()
            plt.show()

    def _compute_paths(self) -> list[list[int]]:
        paths = [[] for _ in range(self.trials)]
        for i in range(self.trials):
            path = [0 for _ in range(self.steps)]
            in_state = np.random.choice(self.state_space, p=self.initial_distribution)
            path[0] = in_state
            pv_state = in_state
            for j in range(1, self.steps):
                cr_state = np.random.choice(
                    self.state_space, p=self.transition_matrix[pv_state]
                )
                path[j] = cr_state
                pv_state = cr_state
            paths[i] = path
        return paths

    def _estimate_final_distribution(self, trials: int = 1_000) -> np.ndarray:
        return self.estimate_distribution_from_initial_state(
            initial_state=np.random.choice(
                self.state_space, p=self.initial_distribution
            ),
            final_epoch=self.steps,
            trials=trials,
        )

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
