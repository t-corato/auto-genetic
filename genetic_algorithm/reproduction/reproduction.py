import numpy as np
from genetic_algorithm.reproduction.crossover import CrossOver
from genetic_algorithm.reproduction.mutation import Mutation
from genetic_algorithm.reproduction.translation import Translation
from genetic_algorithm.reproduction.selection import Selection


class Reproduction:
    def __init__(self, population: np.array = None, crossover_method: str = "single_point_split",
                 prob_crossover: float = 1, hyperparams_values: dict = None, prob_mutation: float = 0.3,
                 prob_translation: float = 0.1, selection_method: str = "roulette_wheel",
                 reproduction_rate: float = 0.2, tournament_size: int = 4):
        self.prob_crossover = prob_crossover
        self.crossover_method = crossover_method
        self.prob_mutation = prob_mutation
        self.prob_translation = prob_translation
        self.reproduction_rate = reproduction_rate
        self.hyperparams_values = hyperparams_values
        self.hyperparams_types = None  # TODO find a way to use the hyperparams types when doing the crossover
        self.population = population
        self.selection_method = selection_method
        self.tournament_size = tournament_size

    def crossover(self, parent_1, parent_2):
        child1, child2 = CrossOver(self.crossover_method).perform_crossover(parent_1=parent_1, parent_2=parent_2)

        return child1, child2

    def mutation(self, child):
        """
        mutation, changing 1 into 0 and the other way around,
        to add more randomness
        """
        mutated_child = Mutation(self.crossover_method, self.hyperparams_values,
                                 self.prob_mutation).perform_mutation(child)

        return mutated_child

    def translation(self, child):
        """
        translate all the element of a gene by one
        """
        translated_child = Translation(self.prob_translation).perform_translation(child)

        return translated_child

    def selection(self):
        """
        selection of a parents using their score
        """
        parents = Selection(self.selection_method, self.population,
                            self.reproduction_rate).perform_selection(tournament_size=self.tournament_size)

        return parents

    def reproduction(self):
        """
        create a generation starting form the previous one, using roulette wheel selection and random choice to select the parents
        """
        parents = self.selection()
        children = []

        for i in range(0, len(parents), 2):
            parent_1, parent_2 = parents[i], parents[i + 1]
            child_1, child_2 = self.crossover(parent_1, parent_2)
            child_1, child_2 = self.mutation(child_1), self.mutation(child_2)
            child_1, child_2 = self.translation(child_1), self.translation(child_2)
            children.append(child_1)
            children.append(child_2)

        return children
