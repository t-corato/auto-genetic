# auto-genetic
Library for automatic optimisation using a genetic algorithm  out-of-the-box

This library is now public and installable via PyPi, with the name auto-genetic.

## How to use 
The main idea of the algorithm is to let the user define a program that has to be optimised afterwards.

## How to define a program
The programs in this case should inherit from auto_genetic.program_base.BaseProgram that is defined as follows:

'''

class BaseProgram(ABC):

    def __init__(self):
        self.hyperparams = None

    def set_program_hyperparams(self, hyperparams: dict):
        self.hyperparams = hyperparams

    @abstractmethod
    def run(self, train_data: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, test_data: pd.DataFrame):
        pass

'''
