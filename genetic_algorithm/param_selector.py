import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseSelector(ABC):
    def __init__(self, gene, data, hyperparams_values):
        self.data = data
        self.gene = gene
        self.hyperparams_values = hyperparams_values

    @abstractmethod
    def select(self):
        pass


class FeatureSelector(BaseSelector):
    def __init__(self, gene, data, hyperparams_values):
        super().__init__(gene, data, hyperparams_values)

    def select(self):
        filterer = np.argwhere(self.gene == 1)
        df_filtered = self.data.iloc[:, filterer.flatten()]
        return df_filtered


class ParameterSelector(BaseSelector):
    def __init__(self, gene, data, hyperparams_values):
        super().__init__(gene, data, hyperparams_values)

    def select(self):
        selected_params = {}
        i = 0
        while i < len(self.gene):
            for key, value in self.hyperparams_values.items():
                selected_params[key] = value[self.gene[i]]
            i += 1

        return selected_params
