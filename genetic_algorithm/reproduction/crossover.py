import numpy as np
import random
from genetic_algorithm.population_initializer.chromosomes import Chromosome
from copy import deepcopy


# TODO handle the fact that parent and children are now classes and not np.array
class CrossOver:
    def __init__(self, crossover_method: str, prob_crossover: float = 1.0) -> None:
        self.crossover_method = crossover_method
        self.prob_crossover = prob_crossover

    def perform_crossover(self, parent_1: Chromosome, parent_2: Chromosome) -> tuple[Chromosome, Chromosome]:

        if self.crossover_method == "single_point_split":
            child_1, child_2 = self._single_split_crossover(parent_1, parent_2)

        elif self.crossover_method == "two_point_split":
            child_1, child_2 = self._two_split_crossover(parent_1, parent_2)

        elif self.crossover_method == "uniform":
            child_1, child_2 = self._uniform_crossover(parent_1, parent_2)
        else:
            raise ValueError("the crossover method selected has not been implemented")

        child_1.fitness = None
        child_2.fitness = None

        return child_1, child_2

    def _single_split_crossover(self, parent_1: Chromosome, parent_2: Chromosome) -> tuple[Chromosome, Chromosome]:

        prob = np.random.rand()

        if prob <= self.prob_crossover:
            # generating the random number to perform crossover
            crossover_point = random.randint(0, len(parent_1.sequence))

            # interchanging the genes
            child1 = np.hstack([parent_1.sequence[: crossover_point], parent_2.sequence[crossover_point:]])
            child2 = np.hstack([parent_2.sequence[: crossover_point], parent_1.sequence[crossover_point:]])

            child_1 = deepcopy(parent_1)
            child_2 = deepcopy(parent_2)

            child_1.sequence = child1
            child_2.sequence = child2

            return child_1, child_2

        else:
            return deepcopy(parent_1), deepcopy(parent_2)

    def _two_split_crossover(self, parent_1: Chromosome, parent_2: Chromosome) -> tuple[Chromosome, Chromosome]:
        prob = np.random.rand()

        if prob <= self.prob_crossover:
            index_1 = random.randint(0, len(parent_1.sequence)//2)
            index_2 = random.randint(index_1, len(parent_1.sequence))

            child1 = np.hstack([parent_1.sequence[: index_1], parent_2.sequence[index_1: index_2],
                               parent_1.sequence[index_2:]])
            child2 = np.hstack([parent_2.sequence[: index_1], parent_1.sequence[index_1: index_2],
                               parent_2.sequence[index_2:]])

            child_1 = deepcopy(parent_1)
            child_2 = deepcopy(parent_2)

            child_1.sequence = child1
            child_2.sequence = child2

            return child_1, child_2

        else:
            return deepcopy(parent_1), deepcopy(parent_2)

    def _uniform_crossover(self, parent_1: Chromosome, parent_2: Chromosome) -> tuple[Chromosome, Chromosome]:
        prob = np.random.rand()

        if prob <= self.prob_crossover:

            index_prob = np.random.rand(len(parent_1.sequence))
            child_1, child_2 = deepcopy(parent_1), deepcopy(parent_2)
            for i in range(len(index_prob)):
                if index_prob[i] <= 0.5:
                    child_1.sequence[i], child_2.sequence[i] = child_2.sequence[i], child_1.sequence[i]

            return child_1, child_2

        else:
            return deepcopy(parent_1), deepcopy(parent_2)






