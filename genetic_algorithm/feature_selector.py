import pandas as pd
import numpy as np


class FeatureSelector:
    def __init__(self, data=None):
        self.data = data

    def feature_select(self, chromosome):
        sequence = chromosome.sequence
        filterer = np.argwhere(sequence == 1)
        df_filtered = self.data.iloc[:, filterer.flatten()]

        return df_filtered

