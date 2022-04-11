import numpy as np
import random
from abc import ABC, abstractmethod
from genetic_algorithm.evaluation.metrics import *


class FitnessFunctionBase(ABC):
    def __init__(self, program, train_data, test_data, target):
        self.train_data = train_data
        self.test_data = test_data
        self.program = program
        self.target = target

    @abstractmethod
    def calculate_fitness(self):
        pass


class RMSEFitness(FitnessFunctionBase):
    def __init__(self, program, train_data, test_data, target):
        super(RMSEFitness, self).__init__(program, train_data, test_data, target)

    def calculate_fitness(self):
        self.program.run(self.train_data)
        predictions = self.program.predict(self.test_data)

        error = self.test_data[self.target] - predictions

        rmse = root_mean_squared_error(error)

        return - rmse


class MAEFitness(FitnessFunctionBase):
    def __init__(self, program, train_data, test_data, target):
        super(MAEFitness, self).__init__(program, train_data, test_data, target)

    def calculate_fitness(self):
        self.program.run(self.train_data)
        predictions = self.program.predict(self.test_data)

        error = self.test_data[self.target] - predictions

        mae = root_mean_squared_error(error)

        return - mae


class SMAPEFitness(FitnessFunctionBase):
    def __init__(self, program, train_data, test_data, target):
        super(SMAPEFitness, self).__init__(program, train_data, test_data, target)

    def calculate_fitness(self):
        self.program.run(self.train_data)
        predictions = self.program.predict(self.test_data)

        s_mape = symmetric_mean_average_percentage_error(self.test_data[self.target], predictions)

        return - s_mape


class MAAPEFitness(FitnessFunctionBase):
    def __init__(self, program, train_data, test_data, target):
        super(MAAPEFitness, self).__init__(program, train_data, test_data, target)

    def calculate_fitness(self):
        self.program.run(self.train_data)
        predictions = self.program.predict(self.test_data)

        s_mape = mean_arctangent_absolute_percentage_error(self.test_data[self.target], predictions)

        return - s_mape










