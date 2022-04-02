import random
import numpy as np
from genetic_algorithm.crossover import CrossOver
from genetic_algorithm.mutation import Mutation
n_factors = 84  # retrieve from size of dataset (to deprecate)

# TODO everything is terribly coded


class GeneticAlgorithm:
    def __init__(self, pop_size=100, number_gen=20, min_value=None, prob_crossover=1,
                 crossover_method="single_point_split", mutation=True, mutation_method="bit_flip", prob_mutation=0.3,
                 prob_translation=0.1, reproduction_rate=0.2):

        self.pop_size = pop_size
        self.number_gen = number_gen
        self.min_value = min_value
        self.prob_crossover = prob_crossover
        self.crossover_method = crossover_method
        self.prob_mutation = prob_mutation
        self.prob_translation = prob_translation
        self.reproduction_rate = reproduction_rate
        self.mutation = mutation
        self.mutation_method = mutation_method
        self.hyperparams_values = None

    def initialize_population(self):
        pass

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

    def translation(self, c1):
        """
        translate all the element of a gene by one
        """
        trans = np.random.rand(1)
        if trans < self.prob_translation:
            c1 = list(c1)
            c1.insert(0, c1.pop())
            c1 = np.array(c1)
        return c1

    def roulette_wheel_selection(self, p):
        """
        selection of a parents using their score
        """
        c = np.cumsum(p)
        r = sum(p) * np.random.rand()
        ind = np.argwhere(r <= c)
        return ind[0][0]

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

    def reproduction(self, pop, scores):
        """
        create a generation starting form the previous one, using roulette wheel selection and random choice to select the parents
        """
        children = []
        for _ in range(int(len(pop) * self.reproduction_rate / 2)):
            p_1 = pop[self.roulette_wheel_selection(scores)]
            p_2 = pop[self.roulette_wheel_selection(scores)]
            c_1, c_2 = self.crossover(p_1, p_2)
            c_1, c_2 = self.mutation(c_1), self.mutation(c_2)
            c_1, c_2 = self.translation(c_1), self.translation(c_2)
            c_1, c_2 = self.swap(c_1), self.swap(c_2)
            children.append(c_1)
            children.append(c_2)
        children = np.array(children)
        return children

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
