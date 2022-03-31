import numpy as np

# flag = np.random.rand(*child.shape) <= self.prob_mutation
# ind = np.argwhere(flag)


# look at how to perform mutation with multiple non-binary arguments


class Mutation:
    def __init__(self, mutation_method):
        self.mutation_method = mutation_method

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

        elif self.mutation_method == "all":
            mutated_gene = self._all_mutation(gene)

        else:
            raise ValueError("the mutation method specified is not implemented")

        return mutated_gene

    def _bit_flip_mutation(self, gene: np.array) -> np.array:
        raise NotImplementedError()

    def _random_resetting_mutation(self, gene: np.array) -> np.array:
        raise NotImplementedError()

    def _swap_mutation(self, gene: np.array) -> np.array:
        raise NotImplementedError()

    def _scramble_mutation(self, gene: np.array) -> np.array:
        raise NotImplementedError()

    def _inversion_mutation(self, gene: np.array) -> np.array:
        raise NotImplementedError()

    def _all_mutation(self, gene: np.array) -> np.array:
        raise NotImplementedError()
