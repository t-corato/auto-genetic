import numpy as np


class FeatureSetter:
    def __init__(self, algo_type: str, feature_num: int):
        self.algo_type = algo_type
        self.feature_num = feature_num

    def set_feature_sequence(self):
        if self.algo_type == "feature_selection":
            sequence = np.random.choice([0, 1], size=self.feature_num, p=[0.5, 0.5])
            return sequence

        else:
            raise ValueError("If the algo_type is not feature_selection, you should not need this method")
