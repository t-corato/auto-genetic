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


class Population(list):
    def __init__(self, n_chromosomes, n_hyperparams, hyperparams_values, features):
        super().__init__()
        pass

    def initialise_pop(self):
        raise NotImplementedError()

    def delete_worst(self):
        raise NotImplementedError()

