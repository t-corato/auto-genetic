import numpy as np
import random


class Selection:
    def __init__(self, selection_method, population, reproduction_rate):
        self.selection_method = selection_method
        self.population = population
        self.reproduction_rate = reproduction_rate
        self.n_parents = self.population * self.reproduction_rate

    def perform_selection(self, tournament_size=None):
        if self.selection_method == "roulette_wheel":
            parents = self._roulette_wheel_selection()
        elif self.selection_method == "stochastic_universal_sampling":
            parents = self._stochastic_universal_sampling()
        elif self.selection_method == "tournament":
            if tournament_size is None:
                raise ValueError("We need a valid tournament_size for this method")
            parents = self._tournament_selection(tournament_size=tournament_size)
        elif self.selection_method == "rank":
            parents = self._rank_selection()
        elif self.selection_method == "random":
            parents = self._random_selection()
        else:
            raise ValueError("The selection method specified is not implemented \n"
                             "Please specify a valid selection method ")

    def _roulette_wheel_selection(self):
        all_parents = []
        while len(all_parents) < self.n_parents:

            population_fitness = sum([chromosome.fitness() for chromosome in self.population])

            chromosome_probabilities = [chromosome.fitness() / population_fitness for chromosome in self.population]

            all_parents.append(np.random.choice(self.population, p=chromosome_probabilities))

        return np.array(all_parents)

    def _stochastic_universal_sampling(self):
        total_fitness = sum([chromosome.fitness() for chromosome in self.population])
        point_distance = total_fitness / self.n_parents
        start_point = random.uniform(0, point_distance)
        points = [start_point + i * point_distance for i in range(self.n_parents)]

        parents = set()
        while len(parents) < self.n_parents:
            random.shuffle(self.population)
            n_rounds = 0
            while n_rounds < len(points) and len(parents) < self.n_parents:
                for gene_num in range(len(self.population)):
                    if self._get_subset_sum(gene_num) > points[n_rounds]:
                        parents.add(self.population[gene_num])
                        break
                n_rounds += 1

        return np.array(parents)

    def _get_subset_sum(self, gene_num):
        subset_sum = 0.0
        i = 0

        while i <= gene_num:
            subset_sum += self.population[i].fitness()
            i += 1
        return subset_sum

    def _tournament_selection(self, tournament_size=4):
        all_parents = []
        while len(all_parents) < self.n_parents:
            parents = random.choices(self.population, k=tournament_size)
            parents = sorted(parents, key=lambda gene: gene.fitness(), reverse=True)
            all_parents.append(parents[0])

        return np.array(all_parents)

    def _rank_selection(self):
        parents = sorted(self.population, key=lambda gene: gene.fitness(), reverse=True)
        best_parents = parents[:self.n_parents]
        return np.array(best_parents)

    def _random_selection(self):
        parents = random.choices(self.population, k=self.n_parents)

        return np.array(parents)
