import numpy as np


class Translation:
    def __init__(self, prob_translation):
        self.prob_translation = prob_translation

    def perform_translation(self, gene):
        prob = np.random.rand()
        translated_gene = gene.deepcopy()
        if prob <= self.prob_translation:
            num_traslated = np.random.randint(0, len(gene))
            translated_gene = np.hstack([translated_gene[-num_traslated:], translated_gene[:-num_traslated]])

        return translated_gene
