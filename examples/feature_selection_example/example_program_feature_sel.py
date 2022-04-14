from genetic_algorithm.program_base import BaseProgram
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class ExampleProgramFeature(BaseProgram):
    def __init__(self):
        super().__init__()
        self.estimator = None

    def run(self, train_data):
        x, y = self._preprocess(train_data)
        if not x.empty:
            self.estimator = RandomForestClassifier()
            self.estimator.fit(x, y)
        else:
            pass

    def predict(self, test_data):
        x, y = self._preprocess(test_data)
        if not x.empty:
            pred = self.estimator.predict(x)
        else:
            pred = np.zeros(shape=y.shape)

        return pred

    @staticmethod
    def _preprocess(df):
        if "Embarked" in df.columns:
            df.loc[df["Embarked"].isna(), "Embarked"] = "S"
        for column in ["Sex", "Embarked", "Pclass"]:
            if column in df.columns:
                temp_df = pd.get_dummies(df[column], drop_first=True, prefix=column)
                df = df.drop(column, axis=1)
                df = pd.concat([df, temp_df], axis=1)

        x = df.drop("Survived", axis=1)
        y = df["Survived"]

        return x, y
