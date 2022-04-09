from genetic_algorithm.program_base import BaseProgram
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


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
