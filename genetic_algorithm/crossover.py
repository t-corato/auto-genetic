import numpy as np
import random


class CrossOver:
    def __init__(self, crossover_method: str) -> None:
        self.crossover_method = crossover_method

    def perform_crossover(self, parent_1: np.array, parent_2: np.array) -> np.array:

        if self.crossover_method == "single_point_split":
            child = self._single_split_crossover(parent_1, parent_2)

        elif self.crossover_method == "two_point_split":
            child = self._two_split_crossover(parent_1, parent_2)

        elif self.crossover_method == "uniform":
            child = self._uniform_crossover(parent_1, parent_2)
        else:
            raise ValueError("the method selected has not been implemented")

        return child

    @staticmethod
    def _single_split_crossover(parent_1: np.array, parent_2: np.array) -> np.array:

        # generating the random number to perform crossover
        crossover_point = random.randint(0, len(parent_1))

        # interchanging the genes
        child = np.hstack(parent_1[: crossover_point], parent_2[crossover_point:])

        return child

    @staticmethod
    def _two_split_crossover(parent_1: np.array, parent_2: np.array) -> np.array:
        raise NotImplementedError()

    @staticmethod
    def _uniform_crossover(parent_1: np.array, parent_2: np.array) -> np.array:
        raise NotImplementedError()
