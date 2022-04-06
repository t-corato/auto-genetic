import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseProgram(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass