import random


class Darwinism:
    def __init__(self, population, reproduction_rate):
        self.population = population
        self.reproduction_rate = reproduction_rate

    def discard_worst_performers(self):
        keep_number = len(self.population) * (1 - self.reproduction_rate)

        self.population.sort(key=lambda chromosome: chromosome.fitness)
        self.population = self.population[:keep_number]
        random.shuffle(self.population)

        return self.population
