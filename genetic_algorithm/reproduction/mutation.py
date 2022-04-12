import numpy as np
import random
from genetic_algorithm.population_initializer.chromosomes import Chromosome
from copy import deepcopy


class Mutation:
    def __init__(self, mutation_method: str, hyperparams_types: dict, prob_mutation: float):
        self.mutation_method = mutation_method
        self.prob_mutation = prob_mutation
        self.hyperparams_types = hyperparams_types
        self.hyperparams_map = None

    def perform_mutation(self, gene: Chromosome) -> Chromosome:
        self.hyperparams_map = deepcopy(gene.hyperparams_map)
        if self.mutation_method == "bit_flip":
            mutated_gene = self._bit_flip_mutation(gene)

        elif self.mutation_method == "random":
            mutated_gene = self._random_resetting_mutation(gene)

        elif self.mutation_method == "swap":
            mutated_gene = self._swap_mutation(gene)

        elif self.mutation_method == "scramble":
            mutated_gene = self._scramble_mutation(gene)

        elif self.mutation_method == "inversion":
            mutated_gene = self._inversion_mutation(gene)

        else:
            raise ValueError("the mutation method specified is not implemented")

        mutated_gene.fitness = None

        return mutated_gene

    def _bit_flip_mutation(self, gene: Chromosome) -> np.array:
        to_be_mutated = np.random.rand(*gene.sequence.shape) <= self.prob_mutation
        mutation_index = np.argwhere(to_be_mutated)
        mutated_gene = deepcopy(gene)

        for i in mutation_index:
            if mutated_gene.sequence[i] == 0:
                mutated_gene.sequence[i] = 1
            else:
                mutated_gene.sequence[i] = 1

        return mutated_gene

    def _random_resetting_mutation(self, gene: Chromosome) -> Chromosome:
        to_be_mutated = np.random.rand(*gene.sequence.shape) <= self.prob_mutation
        mutation_index = np.argwhere(to_be_mutated).flatten()
        mutated_gene = deepcopy(gene)

        for i in mutation_index:
            param = list(self.hyperparams_map.keys())[i]
            if self.hyperparams_types[param] == "continuous":
                choice = self._random_continuous(param)

            else:
                choice = self._random_categorical(param)

            mutated_gene.sequence[i] = choice

        return mutated_gene

    def _swap_mutation(self, gene: Chromosome) -> Chromosome:
        to_be_swapped = np.random.rand(*gene.sequence.shape) <= self.prob_mutation
        swapped_index = np.argwhere(to_be_swapped)
        if swapped_index % 2 == 1:
            swapped_index = np.random.choice(swapped_index, size=len(swapped_index)-1, replace=False)

        mutated_gene = deepcopy(gene)

        for i in range(0, len(swapped_index), 2):
            mutated_gene.sequence[swapped_index[i]], mutated_gene.sequence[swapped_index[i + 1]] = \
                mutated_gene.sequence[swapped_index[i]], mutated_gene.sequence[swapped_index[i + 1]]

        return mutated_gene

    def _scramble_mutation(self, gene: Chromosome) -> Chromosome:
        prob = np.random.rand()
        mutated_child = deepcopy(gene)
        if prob <= self.prob_mutation:
            index_1 = random.randint(0, len(gene.sequence))
            index_2 = random.randint(index_1, len(gene.sequence))

            mutated_child.sequence = np.hstack([mutated_child.sequence[:index_1],
                                               random.shuffle(mutated_child.sequence[index_1:index_2]),
                                               mutated_child.sequence[index_2:]])

        return mutated_child

    def _inversion_mutation(self, gene: Chromosome) -> Chromosome:
        prob = np.random.rand()
        mutated_child = deepcopy(gene)
        if prob <= self.prob_mutation:
            index_1 = random.randint(0, len(gene.sequence))
            index_2 = random.randint(index_1, len(gene.sequence))

            mutated_child.sequence = np.hstack([mutated_child.sequence[:index_1],
                                               mutated_child.sequence[index_1:index_2][::-1],
                                               mutated_child.sequence[index_2:]])

        return mutated_child

    def _random_categorical(self, param):
        values = self.hyperparams_map[param][0]
        choice = np.random.choice(values)

        return  choice

    def _random_continuous(self, param):
        values = self.hyperparams_map[param]
        choice = random.uniform(values[0], values[1])

        return choice
