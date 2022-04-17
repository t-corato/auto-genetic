# auto-genetic
Library for automatic optimisation using a genetic algorithm  out-of-the-box

This library is now public and installable via PyPi, with the name auto-genetic.

## How to use 
The main idea of the algorithm is to let the user define a program that has to be optimised afterwards.

## How to define a program
The programs in this case should inherit from auto_genetic.program_base.BaseProgram that is defined as follows:

    class BaseProgram(ABC):
        """
        Abstract Base Class for the programs, any program has to inherit from this one, or have the same methods, is
        possible to add any additional method, but they need to be run inside either run or predict
        Methods
        -------
        self.set_program_hyperparams(hyperparams): method that is used by each chromosome to set the hyperparameter for
                                                   the program, os it needs to be there
        self.run(train_data): it runs the defined program, all the processes of the program have to be passed here
        self.predict(test_data): it predicts and these predictions are used to calculate the fitness of the program
        """
        def __init__(self):
            self.hyperparams = None

        def set_program_hyperparams(self, hyperparams: dict):
            self.hyperparams = hyperparams

        @abstractmethod
        def run(self, train_data: pd.DataFrame):
            pass

        @abstractmethod
        def predict(self, test_data: pd.DataFrame):
            pass


As we can see the class is abstract and there are 2 abstract methods that need to be implemented, run, where we run the program and pass the selected
hyperparameters or features (according to the type of algorithm we are using) and the other is predict, where we predict the test data that will be used 
to calculate the fitness of the chromosome.

An example of what a new program for an hyperparameter tuning algorithm could look like is this, for the titanic dataset classification problem:


    class ExampleProgram(BaseProgram):
        def __init__(self):
            super().__init__()
            self.estimator = None

        def run(self, train_data):
            x, y = self._preprocess(train_data)
            self.estimator = RandomForestClassifier(**self.hyperparams)
            self.estimator.fit(x, y)

        def predict(self, test_data):
            x, y = self._preprocess(test_data)
            pred = self.estimator.predict(x)

            return pred

        @staticmethod
        def _preprocess(df):
            df = df.drop(["Age", "Cabin", "Name", "Ticket", "PassengerId"], axis=1)
            df.loc[df["Embarked"].isna(), "Embarked"] = "S"
            for column in ["Sex", "Embarked", "Pclass"]:
                temp_df = pd.get_dummies(df[column], drop_first=True, prefix=column)
                df = df.drop(column, axis=1)
                df = pd.concat([df, temp_df], axis=1)

            x = df.drop("Survived", axis=1)
            y = df["Survived"]

            return x, y

Where the run method is doing some preprocessing via self._preprocess and then is fitting the estimator (passing the hyperparameters), while the predict method is predicting on the test data and returning the predictions.

## Fitness Function
Another thing that the user might want to define is the fitness function with which we evaluate the fitness of the chromosomes.
The user has to define the evaluation method of the GeneticAlgorithm in the "set_evaluation_method" Method of the GeneticAlgorithm class.
Here the user can pass a string with the choice of evaluation method that can thake the values: "rmse", "mae", "s_mape", "maape" or "custom".
If the user chooses to define a custom fitness function then he has to define the custom_fitness_function parameter.

    def set_evaluation_method(self, evaluation_method: str, custom_fitness_function=None) -> None:
            """
            Method that sets the fitness function for the program
            Parameters
            ----------
            evaluation_method: str
                               the string that defines the evaluation method that we want to use inside the evaluator,
                               if it's 'custom' the user needs to define the custom_fitness_function
            custom_fitness_function: class
                                     if evaluation_method is custom here we have to have a custom evaluation function
            """

            if self.evaluation_method == "custom" and custom_fitness_function is None:
                raise ValueError("To use a custom evaluation method we need a custom fitness function")

            self.evaluation_method = evaluation_method
            self.custom_fitness_function = custom_fitness_function
            
Here I provide an example of a custom fitness function for the ExampleProgram above, since it's a classification problem the custom fitness function will be the accuracy score

    from sklearn.metrics import accuracy_score
    
    class AccuracyFitness(FitnessFunctionBase):
        def __init__(self, program, train_data, test_data, target):
            super(AccuracyFitness, self).__init__(program, train_data, test_data, target)

        def calculate_fitness(self):
            self.program.run(self.train_data)
            predictions = self.program.predict(self.test_data)

            accuracy = accuracy_score(self.test_data[self.target], predictions)

            return accuracy
            
 ### IMPORTANT: THE ALGORITHM ALWAYS MAXIMISES THE FITNESS FUNCTION, so if you want to minimize you might want to return the negative of the fitness function
        
## Hyperparameters

Then the user needs to define the hyperparameters to be chosen, by passing a list that contains 2 dictionaries to the GeneticAlgorithm class, via the hyperparams_dict parameter. 

The first of the 2 dictionaries needs to define the type of hyperparameter that we are dealing with, if it's categorical or continuous. 
Categorical means that it can take multiple set values from a list, while continuous needs to have 2 values, the minumum and maximum value in the range of the possible values for continuous parameter.
Here is an example for the ExampleProgram defined above, with the hyperparams for the RandomForestClassifier.

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
                        
The second dictionary will define the possible values for our categorical or continuous parameters. The categorical ones needs to have a list with all the possible values for the hyperparameter while the continuous one will have the max and min values.

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

## Run the GeneticAlgorithm
At this point we are ready to define the GeneticAlgorithm and use it for the ExampleProgram:

    import pandas as pd

    from auto_genetic.algorithm import GeneticAlgorithm
    from example_program_hyperparam import ExampleProgram
    from auto_genetic.evaluation.fitness_functions import FitnessFunctionBase
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


    data = pd.read_csv("./train.csv") # the data of the titanic dataset

    genetic_algorithm = GeneticAlgorithm(program=ExampleProgram(), data=data, target_column="Survived",
                                         hyperparams_dict=hyperparams_dict, mutation_method="random",
                                         number_gen=5, pop_size=20, test_size=0.2,
                                         data_split="single", algo_type="hyperparameter_tuning", feature_num=None,
                                         max_fitness_value=None, prob_crossover=1.0, crossover_method="single_point_split",
                                         prob_mutation=0.3, prob_translation=0.1, reproduction_rate=0.2, 
                                         selection_method="roulette_wheel", tournament_size=4)
                                         

    genetic_algorithm.set_evaluation_method(evaluation_method="custom", custom_fitness_function=AccuracyFitness)

    best_chromosome = genetic_algorithm.run()
    
The outputted best chromosome will have an .fitness attribute that contain the fitness and a .hyperparams attribute that contains the set of hyperparams that lead to the best fitness. 

This example program is defined with an Sklearn program but the GeneticAlgorithm is best served by using a more complex custom program, since sklearn 
already has many programs that can optimize the hyperparams in a less Stochastic way.
