import numpy as np
import random


class CrossOver:
    def __init__(self, crossover_method: str) -> None:
        self.crossover_method = crossover_method

    def perform_crossover(self, parent_1: np.array, parent_2: np.array) -> tuple[np.array, np.array]:

        if self.crossover_method == "single_point_split":
            child1, child2 = self._single_split_crossover(parent_1, parent_2)

        elif self.crossover_method == "two_point_split":
            child1, child2 = self._two_split_crossover(parent_1, parent_2)

        elif self.crossover_method == "uniform":
            child1, child2 = self._uniform_crossover(parent_1, parent_2)
        else:
            raise ValueError("the crossover method selected has not been implemented")

        return child1, child2

    @staticmethod
    def _single_split_crossover(parent_1: np.array, parent_2: np.array) -> tuple[np.array, np.array]:

        # generating the random number to perform crossover
        crossover_point = random.randint(0, len(parent_1))

        # interchanging the genes
        child1 = np.hstack(parent_1[: crossover_point], parent_2[crossover_point:])
        child2 = np.hstack(parent_2[: crossover_point], parent_1[crossover_point:])

        return child1, child2

    @staticmethod
    def _two_split_crossover(parent_1: np.array, parent_2: np.array) -> tuple[np.array, np.array]:
        index_1 = random.randint(0, len(parent_1)//2)
        index_2 = random.randint(index_1, len(parent_1))

        child1 = np.hstack(parent_1[: index_1], parent_2[index_1: index_2], parent_1[index_2:])
        child2 = np.hstack(parent_2[: index_1], parent_1[index_1: index_2], parent_2[index_2:])

        return child1, child2

    @staticmethod
    def _uniform_crossover(parent_1: np.array, parent_2: np.array) -> tuple[np.array, np.array]:
        index_prob = np.random.rand(len(parent_1))
        child1, child2 = parent_1.deepcopy(), parent_2.deepcopy()
        for i in range(len(index_prob)):
            if index_prob[i] <= 0.5:
                child1[i], child2[i] = child2[i], child1[i]

        return child1, child2






