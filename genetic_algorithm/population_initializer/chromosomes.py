import numpy as np


class Chromosome:
    def __init__(self, n_hyperparams: int = None, hyperparams_values: dict = None, features: list = None):
        self.fitness = None
        self.fitness_function = None

    def initialise(self):
        raise NotImplementedError()

    def set_fitness_function(self, fitness_function_initialised):
        self.fitness_function = fitness_function_initialised

    def calculate_fitness(self):
        if self.fitness:
            return self.fitness
        else:
            self.fitness = self.fitness_function.calculate_fitness()

