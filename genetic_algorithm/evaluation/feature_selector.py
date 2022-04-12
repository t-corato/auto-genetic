import pandas as pd
import numpy as np


class FeatureSelector:
    def __init__(self, data=None, target=None):
        self.data = data
        self.target = target
        self.x = data.drop(self.target, axis=1)
        self.y = data[self.target]

    def feature_select(self, chromosome):
        sequence = chromosome.sequence
        filterer = np.argwhere(sequence == 1)
        x_filtered = self.x.iloc[:, filterer.flatten()]
        df_filtered = pd.concat([x_filtered, self.y], axis=1)

        chromosome.features = list(df_filtered.columns)

        return df_filtered

