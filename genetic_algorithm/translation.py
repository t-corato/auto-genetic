import numpy as np


class Traslation:
    def __init__(self, prob_traslation):
        self.prob_traslation = prob_traslation

    def perform_traslation(self, gene):
        prob = np.random.rand()
        mutated_gene = gene.copy()
        if prob <= self.prob_traslation:
            num_traslated = np.random.randint(0, len(gene))
            mutated_gene = np.hstack([mutated_gene[-num_traslated:], mutated_gene[:-num_traslated]])

        return mutated_gene
