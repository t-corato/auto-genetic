import random


class Darwinism:
    def __init__(self, population, reproduction_rate):
        self.population = population
        self.reproduction_rate = reproduction_rate
        self.best_chromosome = None

    def discard_worst_performers(self):
        keep_number = int(len(self.population) * (1 - self.reproduction_rate))

        self.population.sort(key=lambda chromosome: chromosome.fitness, reverse=True)
        self.population = self.population[:keep_number]
        self.best_chromosome = self.population[0]
        random.shuffle(self.population)

        return self.population, self.best_chromosome
