import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseSelector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select(self):
        pass
