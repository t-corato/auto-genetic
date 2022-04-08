from genetic_algorithm.evaluation.fitnessfuncsetter import FitnessFuncSetter
from genetic_algorithm.evaluation.feature_selector import FeatureSelector


class Evaluator:
    def __init__(self, program, population, evaluation_method, target, algo_type, train_data, test_data,
                 custom_fitness_function=None):

        self.program = program
        self.population = population
        self.evaluation_method = evaluation_method
        self.train_data = train_data
        self.test_data = test_data
        self.target = target
        self.custom_fitness_function = custom_fitness_function
        self.algo_type = algo_type

    def _set_program_hyperparams(self, chromosome):
        if chromosome.hyperparams is None:
            raise ValueError("There is an issue with the hyperparameters of this chromosome")

        self.program.set_program_hyperparams(chromosome.hyperparams)

    def _filter_data(self, chromosome):
        selector_train = FeatureSelector(self.train_data)
        selector_test = FeatureSelector(self.test_data)

        train_data, test_data = selector_train.feature_select(chromosome), selector_test.feature_select(
            chromosome)

        return train_data, test_data

    def evaluate_chromosome(self, chromosome):
        """
        evaluate a chromosome using the TCN
        """
        if self.algo_type == "feature_selection":
            train_data, test_data = self._filter_data(chromosome)
            func_setter = FitnessFuncSetter(self.evaluation_method, self.program, train_data, test_data,
                                            self.target)
        elif self.algo_type == "hyperparameter_tuning":
            self._set_program_hyperparams(chromosome)
            func_setter = FitnessFuncSetter(self.evaluation_method, self.program, self.train_data, self.test_data,
                                            self.target)

        else:
            raise ValueError("Please specify a valid algo_type")

        fitness_function = func_setter.set_fitness_func(
            custom_fitness_function=self.custom_fitness_function).get_fitness_func()
        chromosome.set_fitness_function(fitness_function)

        chromosome.calculate_fitness()

    def evaluate_generation(self):
        for chromosome in self.population:
            self.evaluate_chromosome(chromosome)
