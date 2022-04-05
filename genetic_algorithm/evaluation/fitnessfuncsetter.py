from genetic_algorithm.evaluation.fitness_functions import *


class FitnessFuncSetter:
    def __init__(self, evaluation_method, program, train_data, test_data, target):
        self.evaluation_method = evaluation_method
        self.program = program
        self.train_data = train_data
        self.test_data = test_data
        self.target = target
        self.fitness_func = None

    def set_fitness_func(self, custom_fitness_function=None):
        if self.evaluation_method == "custom":
            if custom_fitness_function is None:
                raise ValueError("to use the custom method you need to implement a custom fitness function"
                                 "inheriting the FitnessFunction class or (at your own risk) by implementing it freely")

            self.fitness_func = custom_fitness_function(self.program, self.train_data, self.test_data, self.target)

        elif self.evaluation_method == "rmse":
            self.fitness_func = RMSEFitness(self.program, self.train_data, self.test_data, self.target)

        elif self.evaluation_method == "mae":
            self.fitness_func = MAEFitness(self.program, self.train_data, self.test_data, self.target)

        elif self.evaluation_method == "s_mape":
            self.fitness_func = SMAPEFitness(self.program, self.train_data, self.test_data, self.target)

        elif self.evaluation_method == "maape":
            self.fitness_func = MAAPEFitness(self.program, self.train_data, self.test_data, self.target)

        else:
            raise ValueError("The selected evaluation method is not available, choose one of the available ones or "
                             "implement your own by choosing custom")

