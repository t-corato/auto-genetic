from genetic_algorithm.evaluation.fitnessfuncsetter import FitnessFuncSetter


class Evaluator:
    def __init__(self, program, population, evaluation_method, train_data, test_data, target,
                 custom_fitness_function=None):

        self.program = program
        self.population = population
        self.evaluation_method = evaluation_method
        self.train_data = train_data
        self.test_data = test_data
        self.target = target
        self.custom_fitness_function = custom_fitness_function

    def evaluate_chromosome(self, chromosome):
        """
        evaluate a chromosome using the TCN
        """
        setter = FitnessFuncSetter(self.evaluation_method, self.program, self.train_data, self.test_data,
                                   self.target)
        fitness_function = setter.set_fitness_func(
            custom_fitness_function=self.custom_fitness_function).get_fitness_func()
        chromosome.set_fitness_function(fitness_function)

        chromosome.calculate_fitness()

    def evaluate_generation(self):
        for chromosome in self.population:
            self.evaluate_chromosome(chromosome)
