from tqdm import tqdm
import pandas as pd

from genetic_algorithm.reproduction.reproduction import Reproduction
from genetic_algorithm.evaluation.evaluator import Evaluator
from genetic_algorithm.darwinism import Darwinism
from genetic_algorithm.population_initializer.population import PopulationInitializer
from sklearn.model_selection import train_test_split


class GeneticAlgorithm:
    def __init__(self, program, data: pd.DataFrame, target_column: str, test_size: float = 0.2,
                 data_split: str = "single", algo_type: str = "hyperparameter_tuning", pop_size: int = 100,
                 number_gen: int = 20, hyperparams_dict=None, feature_num=None,  max_fitness_value: float = None,
                 prob_crossover: float = 1.0, crossover_method: str = "single_point_split",
                 mutation_method: str = "bit_flip", prob_mutation: float = 0.3, prob_translation: float = 0.1,
                 reproduction_rate: float = 0.2, selection_method: str = "roulette_wheel", tournament_size: int = 4):

        self.algo_type = algo_type
        self.pop_size = pop_size
        self.number_gen = number_gen
        self.max_fitness_value = max_fitness_value
        self.prob_crossover = prob_crossover
        self.crossover_method = crossover_method
        self.prob_mutation = prob_mutation
        self.prob_translation = prob_translation
        self.reproduction_rate = reproduction_rate
        self.mutation_method = mutation_method
        self.hyperparams_dict = hyperparams_dict
        if self.hyperparams_dict:
            self.hyperparams_names = self.hyperparams_dict[0].keys()
            self.hyperparams_values = self.hyperparams_dict[1]
            self.hyperparams_types = self.hyperparams_dict[0]
        else:
            self.hyperparams_names = None
            self.hyperparams_values = None
            self.hyperparams_types = None
        self.population = None
        self.selection_method = selection_method
        self.best_chromosome = None
        self.tournament_size = tournament_size
        self.evaluation_method = None
        self.program = program
        self.target = target_column
        self.train_data = None
        self.test_data = None
        self.custom_fitness_function = None
        self.feature_num = feature_num
        self.data = data
        self.test_size = test_size
        self.data_split = data_split

    def initialize_population(self):
        self.population = PopulationInitializer(self.pop_size, self.algo_type, hyperparams_dict=self.hyperparams_dict,
                                                feature_num=self.feature_num).initialize_population()

    def split_data(self):
        if self.data_split == "single":
            if self.train_data is None and self.test_data is None:
                self.train_data, self.test_data = train_test_split(self.data, test_size=self.test_size)
        elif self.data_split == "multiple":
            self.train_data, self.test_data = train_test_split(self.data, test_size=self.test_size)

        else:
            raise ValueError("The type of data split specified does not exist")

    def set_evaluation_method(self, evaluation_method, custom_fitness_function=None):

        if self.evaluation_method == "custom" and custom_fitness_function is None:
            raise ValueError("To use a custom evaluation method we need a custom fitness function")

        self.evaluation_method = evaluation_method
        self.custom_fitness_function = custom_fitness_function

    def evaluate_generation(self):
        evaluator = Evaluator(self.program, self.population, self.evaluation_method, self.target, self.algo_type,
                              self.train_data, self.test_data, self.custom_fitness_function)

        evaluator.evaluate_generation()

    def reproduction(self):

        rep_function = Reproduction(population=self.population, crossover_method=self.crossover_method,
                                    prob_crossover=self.prob_crossover, hyperparams_types=self.hyperparams_types,
                                    mutation_method=self.mutation_method,
                                    prob_mutation=self.prob_mutation, prob_translation=self.prob_translation,
                                    selection_method=self.selection_method, reproduction_rate=self.reproduction_rate,
                                    tournament_size=self.tournament_size)
        children = rep_function.reproduction()

        return children

    def darwinism(self):
        """
        removes the worst elements from a population, to make space for the children
        """

        pop_selector = Darwinism(self.population, self.reproduction_rate)
        self.population, self.best_chromosome = pop_selector.discard_worst_performers()

    def run(self):

        """
        run the genetic algorithm for n generation with m genes, storing the best score and the best gene
        """
        self.initialize_population()

        if self.evaluation_method is None:
            raise ValueError("Remember to set the evaluation method before running the algorithm")

        for _ in tqdm(range(self.number_gen)):

            self.split_data()
            self.evaluate_generation()
            children = self.reproduction()

            self.darwinism()

            self.population = self.population + children

            if self.max_fitness_value:
                if self.best_chromosome.fitness > self.max_fitness_value:
                    break

        return self.best_chromosome
