from genetic_algorithm.evaluation.fitness_functions import *


class Evaluator:
    def __init__(self, evaluation_method, program, train_data, test_data, target):
        self.evaluation_method = evaluation_method
        self.program = program
        self.train_data = train_data
        self.test_data = test_data
        self.target = target
        self.fitness_func = None

    def perform_evaluation(self, custom_fitness_function=None):
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
