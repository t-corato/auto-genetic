import pandas as pd

from genetic_algorithm.algorithm import GeneticAlgorithm
from example_program_hyperparam import ExampleProgram
from genetic_algorithm.evaluation.fitness_functions import FitnessFunctionBase
from sklearn.metrics import accuracy_score

hyperparam_values = {"n_estimators": [50, 100, 200],
                     "criterion": ['gini', 'entropy'],
                     "max_depth": [None, 2, 3, 5],
                     "min_samples_split": [0.01, 0.5],
                     "min_samples_leaf": [1, 2, 4],
                     "min_weight_fraction_leaf": [0.0, 0.5],
                     "max_features": ['auto', 'sqrt', 'log2'],
                     "max_leaf_nodes": [None, 2, 3],
                     "min_impurity_decrease": [0.0, 0.3],
                     "oob_score": [False, True],
                     "max_samples": [None, 100, 200]}

hyperparam_types = {"n_estimators": "categorical",
                    "criterion": "categorical",
                    "max_depth": "categorical",
                    "min_samples_split": "continuous",
                    "min_samples_leaf": "categorical",
                    "min_weight_fraction_leaf": "continuous",
                    "max_features": "categorical",
                    "max_leaf_nodes": "categorical",
                    "min_impurity_decrease": "continuous",
                    "oob_score": "categorical",
                    "max_samples": "categorical"}

hyperparams_dict = [hyperparam_types, hyperparam_values]


class AccuracyFitness(FitnessFunctionBase):
    def __init__(self, program, train_data, test_data, target):
        super(AccuracyFitness, self).__init__(program, train_data, test_data, target)

    def calculate_fitness(self):
        self.program.run(self.train_data)
        predictions = self.program.predict(self.test_data)

        accuracy = accuracy_score(self.test_data[self.target], predictions)

        return accuracy


data = pd.read_csv("/Users/tommasocorato/Desktop/train.csv")

genetic_algorithm = GeneticAlgorithm(program=ExampleProgram(), data=data, target_column="Survived",
                                     hyperparams_dict=hyperparams_dict, mutation_method="random",
                                     number_gen=5, pop_size=20)

genetic_algorithm.set_evaluation_method(evaluation_method="custom", custom_fitness_function=AccuracyFitness)

best_chromosome = genetic_algorithm.run()
