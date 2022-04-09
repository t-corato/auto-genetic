import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseProgram(ABC):
    def __init__(self):
        self.hyperparams = None

    def set_program_hyperparams(self, hyperparams: dict):
        self.hyperparams = hyperparams

    @abstractmethod
    def run(self, train_data):
        pass

    @abstractmethod
    def predict(self, test_data):
        pass
