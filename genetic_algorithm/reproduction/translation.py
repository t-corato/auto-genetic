import numpy as np
from genetic_algorithm.population_initializer.chromosomes import Chromosome
from copy import deepcopy


class Translation:
    def __init__(self, prob_translation):
        self.prob_translation = prob_translation

    def perform_translation(self, gene: Chromosome):
        prob = np.random.rand()
        translated_gene = deepcopy(gene)
        if prob <= self.prob_translation:
            num_traslated = np.random.randint(0, len(gene.sequence))
            translated_gene.sequence = np.hstack([translated_gene.sequence[-num_traslated:],
                                                  translated_gene.sequence[:-num_traslated]])

        translated_gene.fitness = None

        return translated_gene
