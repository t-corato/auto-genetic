# auto-genetic
Library for automatic optimisation using a genetic algorithm  out-of-the-box

This library is now public and installable via PyPi, with the name auto-genetic.

## How to use 
The main idea of the algorithm is to let the user define a program that has to be optimised afterwards.

## How to define a program
The programs in this case should inherit from auto_genetic.program_base.BaseProgram that is defined as follows:

'''

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


'''

As we can see the class is abstract and there are 2 abstract methods that need to be implemented, run, where we run the program and pass the selected
hyperparameters or features (according to the type of algorithm we are using) and the other is predict, where we predict the test data that will be used 
to calculate the fitness of the chromosome.

An example of what a new program for an hyperparameter tuning algorithm could look like is this, for the titanic dataset classification problem:

'''

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
'''

Where the run method is doing some preprocessing via self._preprocess and then is fitting the estimator (passing the hyperparameters), while the predict method is predicting on the test data and returning the predictions 
