# evaluator should run the program and get the metrics for each individual child, we can implement several ways but
# there should be a way to implement a custom fitness function

class Evaluator:
    def __init__(self, evaluation_method):
        self.evaluation_method = evaluation_method

    def evaluate(self, custom_fitness_function=None):
        if self.evaluation_method == "custom":
            if custom_fitness_function is None:
                raise ValueError("to use the custom method you need to implement a custom fitness function"
                                 "inheriting the FitnessFunction class or (at your own risk) by implementing it freely")

        elif self.evaluation_method == "rmse":
            raise NotImplementedError()

        else:
            raise ValueError("The selected evaluation method is not available, choose one of the available ones or "
                             "implement your own by choosing custom")
