import random
import numpy as np
from genetic_algorithm.crossover import CrossOver
from genetic_algorithm.mutation import Mutation
from genetic_algorithm.translation import Translation
from genetic_algorithm.selection import Selection


class GeneticAlgorithm:
    def __init__(self, algo_type: str = "hyperparameter_tuning", pop_size: int = 100, number_gen: int = 20,
                 min_fitness_value: float = None, prob_crossover: int = 1, crossover_method: str = "single_point_split",
                 perform_mutation: bool = True, mutation_method: str = "bit_flip", prob_mutation: float = 0.3,
                 prob_translation: float = 0.1, reproduction_rate: float = 0.2,
                 selection_method: str = "roulette_wheel", tournament_size: int = 4):

        self.algo_type = algo_type
        self.pop_size = pop_size
        self.number_gen = number_gen
        self.min_fitness_value = min_fitness_value
        self.prob_crossover = prob_crossover
        self.crossover_method = crossover_method
        self.prob_mutation = prob_mutation
        self.prob_translation = prob_translation
        self.reproduction_rate = reproduction_rate
        self.perform_mutation = perform_mutation
        self.mutation_method = mutation_method
        self.hyperparams_values = None
        self.population = None
        self.selection_method = selection_method
        self.best_gene = None
        self.tournament_size = tournament_size
        self.evaluation_function = None

    def initialize_population(self):
        if self.algo_type == "hyperparameter_tuning":
            pass
        elif self.algo_type == "feature_selection":
            pass
        else:
            raise ValueError('the only algo_type acceptable are "hyperparameter_tuning" and "feature_selection"'
                             'please select one of the 2')

    # TODO CrossOver and Mutation should be defined in the __init__ and then we just call them (?)
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
        children = np.array(children)
        return children

######################################################################################
    # TODO implement params select and feature select proper
    def feature_select(self, X, gene):
        """
        deactivate the columns of the dataframe where the gene is 0
        """
        feature_index = []
        for i in range(len(gene)):
            if gene[i] == 1:
                feature_index.append(i)
        df_filter = X[:, feature_index]
        return df_filter

    def set_evaluation_function(self, evaluation_function):
        self.evaluation_function = evaluation_function

    def evaluate(self):
        """
        evaluate a cromosome using the TCN
        """

        raise NotImplementedError()

    def generation_eval(self, pop):
        """
        evaluate all the scores of a generation, returns all the scores, the best score and the gene that gave the best score
        """
        scores = []
        best_score = 0
        best_set = []
        for i in range(len(pop)):
            score = self.evaluate()
            scores.append(score)
            if score > best_score:
                best_score = score
                best_set = pop[i]
        scores = np.array(scores)
        return scores, best_score, best_set

    def darwin(self, pop, scores):
        """
        removes the worst elements from a population, to make space for the children
        """
        scores = list(scores)
        pop = list(pop)
        for _ in range(int(len(pop) * self.reproduction_rate)):
            x = scores.index(sorted(scores)[0])
            pop.pop(x)
            scores.pop(x)
        pop = np.array(pop)
        scores = np.array(scores)
        return pop, scores

    def GA(self, pop=100, gen=20, n_factors=84):

        """
        run the genetic algorithm for n generation with m genes, storing the best score and the best gene
        """
        parents = []
        for i in range(pop):
            i = np.random.choice([0, 1], size=(n_factors,), p=[1. / 3, 2. / 3])
            parents.append(i)
        parents = np.array(parents)

        best_score = 0
        best_set = []
        scores, gen_best_score, gen_best_set = self.generation_eval(parents)
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_set = gen_best_set
            print(f"Best score gen 1: {best_score}")
        print("Finished generation: 1")
        for i in range(gen - 1):
            children = self.reproduction(parents, scores)
            child_scores, gen_best_score, gen_best_set = self.generation_eval(children)
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_set = gen_best_set
                print(f"Best score gen {i + 2}: {best_score}")
            print(f"Finished generation: {i + 2}")

            parents, scores = self.darwin(parents, scores)
            parents = np.concatenate((parents, children))
            scores = np.concatenate((scores, child_scores))

        return best_score, best_set
