import numpy as np


def root_mean_squared_error(error):
    return np.sqrt((error ** 2).mean())


def mean_absolute_error(error):
    return abs(error).mean()


def symmetric_mean_average_percentage_error(y_true, y_pred):
    errors = 2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)) * 100
    return errors.mean()


def mean_arctangent_absolute_percentage_error(y_true, y_pred):
    aape = np.arctan(abs((y_true - y_pred) / y_true))
    return aape.mean()
