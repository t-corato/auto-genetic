import numpy as np
from genetic_algorithm.reproduction.reproduction import Reproduction
from genetic_algorithm.evaluation.evaluator import Evaluator


class GeneticAlgorithm:
    def __init__(self, program, target_column, algo_type: str = "hyperparameter_tuning", pop_size: int = 100, number_gen: int = 20,
                 min_fitness_value: float = None, prob_crossover: float = 1.0, crossover_method: str = "single_point_split",
                 mutation_method: str = "bit_flip", prob_mutation: float = 0.3,
                 prob_translation: float = 0.1, reproduction_rate: float = 0.2,
                 selection_method: str = "roulette_wheel", tournament_size: int = 4):

        self.algo_type = algo_type
        self.pop_size = pop_size
        self.number_gen = number_gen
        self.min_fitness_value = min_fitness_value
        self.prob_crossover = prob_crossover
        self.crossover_method = crossover_method
        self.prob_mutation = prob_mutation
        self.prob_translation = prob_translation
        self.reproduction_rate = reproduction_rate
        self.mutation_method = mutation_method
        self.hyperparams_names = None
        self.hyperparams_values = None
        self.hyperparams_types = None
        self.population = None
        self.selection_method = selection_method
        self.best_gene = None
        self.tournament_size = tournament_size
        self.evaluation_method = None
        self.program = program
        self.target = target_column
        self.train_data = None
        self.test_data = None
        self.custom_fitness_function = None

    def initialize_population(self):
        if self.algo_type == "hyperparameter_tuning":
            pass
        elif self.algo_type == "feature_selection":
            pass
        else:
            raise ValueError('the only algo_type acceptable are "hyperparameter_tuning" and "feature_selection"'
                             'please select one of the 2')

    def get_hyperparameters(self, hyperparams_dict):
        if self.algo_type == "hyperparameter_tuning":
            self.hyperparams_names = hyperparams_dict[0].keys()
            self.hyperparams_values = hyperparams_dict[1]
            self.hyperparams_types = hyperparams_dict[0]

        else:
            raise ValueError("If the algo_type is not hyperparameter_tuning, you should not need this method")

    def reproduction(self):

        rep_function = Reproduction(population=self.population, crossover_method=self.crossover_method,
                                    prob_crossover=self.prob_crossover, hyperparams_values=self.hyperparams_values,
                                    prob_mutation=self.prob_mutation, prob_translation=self.prob_translation,
                                    selection_method=self.selection_method, reproduction_rate=self.reproduction_rate,
                                    tournament_size=self.tournament_size)
        children = rep_function.reproduction()

        return children

    def set_evaluation_method(self, evaluation_method, custom_fitness_function=None):

        if self.evaluation_method == "custom" and custom_fitness_function is None:
            raise ValueError("To use a ")

        self.evaluation_method = evaluation_method
        self.custom_fitness_function = custom_fitness_function

    def evaluate_generation(self):
        evaluator = Evaluator(self.program, self.population, self.evaluation_method, self.train_data, self.test_data,
                              self.target, self.custom_fitness_function)

        evaluator.evaluate_generation()

################################################################################################################

    def darwin(self, pop, scores):
        """
        removes the worst elements from a population, to make space for the children
        """
        scores = list(scores)
        pop = list(pop)
        for _ in range(int(len(pop) * self.reproduction_rate)):
            x = scores.index(sorted(scores)[0])
            pop.pop(x)
            scores.pop(x)
        pop = np.array(pop)
        scores = np.array(scores)
        return pop, scores

    def GA(self, pop=100, gen=20, n_factors=84):

        """
        run the genetic algorithm for n generation with m genes, storing the best score and the best gene
        """
        parents = []
        for i in range(pop):
            i = np.random.choice([0, 1], size=(n_factors,), p=[1. / 3, 2. / 3])
            parents.append(i)
        parents = np.array(parents)

        best_score = 0
        best_set = []
        scores, gen_best_score, gen_best_set = self.generation_eval(parents)
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_set = gen_best_set
            print(f"Best score gen 1: {best_score}")
        print("Finished generation: 1")
        for i in range(gen - 1):
            children = self.reproduction(parents, scores)
            child_scores, gen_best_score, gen_best_set = self.generation_eval(children)
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_set = gen_best_set
                print(f"Best score gen {i + 2}: {best_score}")
            print(f"Finished generation: {i + 2}")

            parents, scores = self.darwin(parents, scores)
            parents = np.concatenate((parents, children))
            scores = np.concatenate((scores, child_scores))

        return best_score, best_set
