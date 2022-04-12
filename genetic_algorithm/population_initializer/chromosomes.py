from genetic_algorithm.population_initializer.hyperparams_setter import HyperParamsSetter
from genetic_algorithm.population_initializer.feature_setter import FeatureSetter


class Chromosome:
    def __init__(self, algo_type, hyperparams_dict=None, feature_num=None):
        self.fitness = None
        self.fitness_function = None
        self.algo_type=algo_type
        if self.algo_type == "hyperparameter_tuning" and hyperparams_dict is None:
            raise ValueError("We don't have any hyerparameters to tune!!")

        elif self.algo_type == "feature_selection" and feature_num is None:
            raise ValueError("What's the number of features you want to tune?")

        self.hyperparams_dict = hyperparams_dict
        self.feature_num = feature_num
        self.hyperparams = None
        self.sequence = None
        self.hyperparams_map = None
        self.features = None

    def initialize(self):
        if self.algo_type == "hyperparameter_tuning":
            self._initialize_hyperparams()
        elif self.algo_type == "feature_selection":
            self._initialize_feature()

        return self

    def set_fitness_function(self, fitness_function_initialised):
        self.fitness_function = fitness_function_initialised

    def calculate_fitness(self):
        if self.fitness:
            return self.fitness
        else:
            self.fitness = self.fitness_function.calculate_fitness()

    def _initialize_hyperparams(self):
        setter = HyperParamsSetter(self.hyperparams_dict, self.algo_type).get_hyperparameters()
        self.sequence = setter.convert_hyperparams_values()
        self.hyperparams, self.hyperparams_map = setter.get_program_hyperparams()

    def _initialize_feature(self):
        setter = FeatureSetter(self.algo_type, self.feature_num)
        self.sequence = setter.set_feature_sequence()
