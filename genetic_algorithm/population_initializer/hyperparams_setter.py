import numpy as np
import random


class HyperParamsSetter:
    def __init__(self, hyperparams_dict, algo_type):
        self.hyperparams_dict = hyperparams_dict
        self.algo_type = algo_type
        self.hyperparams_values = None
        self.hyperparams_names = None
        self.hyperparams_types = None
        self.hyperparams_map = None
        self.sequence = None

    def get_hyperparameters(self):
        if self.algo_type == "hyperparameter_tuning":
            self.hyperparams_names = self.hyperparams_dict[0].keys()
            self.hyperparams_values = self.hyperparams_dict[1]
            self.hyperparams_types = self.hyperparams_dict[0]

        else:
            raise ValueError("If the algo_type is not hyperparameter_tuning, you should not need this method")
        return self

    def convert_hyperparams_values(self):
        self.hyperparams_map = {}
        sequence = []
        for param, values in self.hyperparams_values.items():
            if self.hyperparams_types[param] == "categorical":
                choice = self._set_categorical(param, values)
                sequence.append(choice)
            elif self.hyperparams_types[param] == "continuous":
                choice = self._set_continuous(param, values)
                sequence.append(choice)

            else:
                raise ValueError(f"The parameter: {param}, is neither categorical nor continuous, check the "
                                 f"hyperparameter types and restart")
        self.sequence = sequence

        return np.array(sequence)

    def get_program_hyperparams(self):
        i = 0
        hyperparams = {}
        for name in self.hyperparams_names:
            index = self.sequence[i]
            if self.hyperparams_types[name] == "categorical":
                parameter = self.hyperparams_map[name][1][index]

            else:
                parameter = index

            hyperparams[name] = parameter
            i += 1
        return hyperparams, self.hyperparams_map

    def _set_continuous(self, param, values):

        self.hyperparams_map[param] = values
        choice = random.uniform(values[0], values[1])

        return choice

    def _set_categorical(self, param, values):
        choices = list(range(len(values)))
        self.hyperparams_map[param] = [choices, values]
        choice = np.random.choice(choices)

        return choice
