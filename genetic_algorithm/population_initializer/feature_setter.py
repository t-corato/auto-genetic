import random
import numpy as np


class FeatureSetter:
    def __init__(self, algo_type, feature_num):
        self.algo_type = algo_type
        self.feature_num = feature_num

    def set_feature_sequence(self):
        if self.algo_type == "feature_selection":
            sequence = np.random.randint(0, 2, size=self.feature_num)
            return sequence

        else:
            raise ValueError("If the algo_type is not feature_selection, you should not need this method")
