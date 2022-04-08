import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


class DataProcessor:
    def __init__(self, data: pd.DataFrame, test_size: float = 0.2):
        self.data = data
        self.test_size = test_size
        self.train_data = None
        self.test_data = None
        self.df_filtered = self.data

    def feature_select(self, sequence):
        filterer = np.argwhere(sequence == 1)
        self.df_filtered = self.data.iloc[:, filterer.flatten()]

    def train_test_split(self):
        self.train_data, self.test_data = train_test_split(self.df_filtered, test_size=self.test_size)

