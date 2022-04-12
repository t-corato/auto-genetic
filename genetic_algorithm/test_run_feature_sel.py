import pandas as pd

from genetic_algorithm.algorithm import GeneticAlgorithm
from example_program_feature_sel import ExampleProgramFeature
from genetic_algorithm.evaluation.fitness_functions import FitnessFunctionBase
from sklearn.metrics import accuracy_score

features =["Pclass", "Sex", "SibSp", "Parch", "Fare", 'Embarked']
n_features = len(features)


class AccuracyFitness(FitnessFunctionBase):
    def __init__(self, program, train_data, test_data, target):
        super(AccuracyFitness, self).__init__(program, train_data, test_data, target)

    def calculate_fitness(self):
        self.program.run(self.train_data)
        predictions = self.program.predict(self.test_data)

        accuracy = accuracy_score(self.test_data[self.target], predictions)

        return accuracy


data = pd.read_csv("/Users/tommasocorato/Desktop/train.csv")
data = data.drop(["Age", "Cabin", "Name", "Ticket", "PassengerId"], axis=1)

genetic_algorithm = GeneticAlgorithm(program=ExampleProgramFeature(), data=data, target_column="Survived",
                                     feature_num=n_features, mutation_method="bit_flip", algo_type="feature_selection",
                                     number_gen=5, pop_size=20)

genetic_algorithm.set_evaluation_method(evaluation_method="custom", custom_fitness_function=AccuracyFitness)

best_chromosome = genetic_algorithm.run()
