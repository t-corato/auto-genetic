from genetic_algorithm.population_initializer.chromosomes import Chromosome


class PopulationInitializer:
    def __init__(self, pop_size, algo_type, hyperparams_dict=None, feature_num=None):
        self.population = []
        self.algo_type = algo_type
        self.hyperparams_dict = hyperparams_dict
        self.feature_num = feature_num
        self.pop_size = pop_size

    def initialize_population(self):
        for _ in range(self.pop_size):
            chromosome = Chromosome(self.algo_type, hyperparams_dict=self.hyperparams_dict,
                                    feature_num=self.feature_num).initialize()

            self.population.append(chromosome)

        return self.population
