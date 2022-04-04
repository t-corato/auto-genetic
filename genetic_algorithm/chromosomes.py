import numpy as np


class Chromosome:
    def __init__(self, n_hyperparams, hyperparams_values, features):
        self.fitness = None

    def initialise(self):
        raise NotImplementedError()

    def fitness(self):
        if self.fitness:
            return self.fitness
        else:
            raise NotImplementedError()


class Population:
    def __init__(self, n_chromosomes, n_hyperparams, hyperparams_values, features):
        pass

    def initialise_pop(self):
        raise NotImplementedError()
