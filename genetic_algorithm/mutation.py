import numpy as np
import random

# flag = np.random.rand(*child.shape) <= self.prob_mutation
# ind = np.argwhere(flag)


# look at how to perform mutation with multiple non-binary arguments


class Mutation:
    def __init__(self, mutation_method: str, hyperparams_values: dict, prob_mutation: float):
        self.mutation_method = mutation_method
        self.hyperparams_values = hyperparams_values
        self.prob_mutation = prob_mutation

    def perform_mutation(self, gene: np.array) -> np.array:
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

        return mutated_gene

    def _bit_flip_mutation(self, gene: np.array) -> np.array:
        to_be_mutated = np.random.rand(*gene.shape) <= self.prob_mutation
        mutation_index = np.argwhere(to_be_mutated)
        mutated_gene = gene.copy()

        for i in mutation_index:
            if mutated_gene[i] == 0:
                mutated_gene[i] = 1
            else:
                mutated_gene[i] = 1

        return mutated_gene

    def _random_resetting_mutation(self, gene: np.array) -> np.array:
        to_be_mutated = np.random.rand(*gene.shape) <= self.prob_mutation
        mutation_index = np.argwhere(to_be_mutated)
        mutated_gene = gene.copy()

        for i in mutation_index:
            mutated_gene[i] = random.choice(self.hyperparams_values[i])

        return mutated_gene

    def _swap_mutation(self, gene: np.array) -> np.array:
        to_be_swapped = np.random.rand(*gene.shape) <= self.prob_mutation
        swapped_index = np.argwhere(to_be_swapped)
        if swapped_index % 2 == 1:
            swapped_index = np.random.choice(swapped_index, size=len(swapped_index)-1, replace=False)

        mutated_gene = gene.copy()

        for i in range(0, len(swapped_index), 2):
            mutated_gene[swapped_index[i]], mutated_gene[swapped_index[i + 1]] = \
                mutated_gene[swapped_index[i]], mutated_gene[swapped_index[i + 1]]

        return mutated_gene

    def _scramble_mutation(self, gene: np.array) -> np.array:
        prob = np.random.rand()
        mutated_child = gene.copy()
        if prob <= self.prob_mutation:
            index_1 = random.randint(0, len(gene))
            index_2 = random.randint(index_1, len(gene))

            mutated_child = np.hstack(mutated_child[:index_1], random.shuffle(mutated_child[index_1:index_2]),
                                      mutated_child[index_2:])

        return mutated_child

    def _inversion_mutation(self, gene: np.array) -> np.array:
        prob = np.random.rand()
        mutated_child = gene.copy()
        if prob <= self.prob_mutation:
            index_1 = random.randint(0, len(gene))
            index_2 = random.randint(index_1, len(gene))

            mutated_child = np.hstack(mutated_child[:index_1], mutated_child[index_1:index_2][::-1],
                                      mutated_child[index_2:])

        return mutated_child
