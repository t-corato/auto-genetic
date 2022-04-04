import numpy as np


class Chromosome:
    def __init__(self, n_hyperparams, hyperparams_values):
        pass

    def initialise(self):
        raise NotImplementedError()

    def fitness(self):
        raise NotImplementedError()
