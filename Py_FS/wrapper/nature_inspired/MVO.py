"""
Programmer: Soumitri Chattopadhyay
Date of Development: 27/07/2021
This code has been developed according to the procedures mentioned in the following research article:
"Mirjalili, S. and Mirjalili, S.M. and Hatamlou, A. (2016). Multi-Verse Optimizer: a nature-inspired algorithm for global optimization"

"""

import math
import numpy as np

from Py_FS.wrapper.nature_inspired.algorithm import Algorithm
from Py_FS.wrapper.nature_inspired._utilities_test import compute_fitness, sort_agents, compute_accuracy

from sklearn import datasets
from sklearn import preprocessing


class MVO(Algorithm):

    # Multi-Verse Optimizer
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of agents                                              #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #
    #   obj_function: the function to maximize while doing feature selection      #
    #   trans_function_shape: shape of the transfer function used                 #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################

    def __init__(self,
                 num_agents,
                 max_iter,
                 train_data,
                 train_label,
                 trans_func_shape='s',
                 save_conv_graph=False,
                 seed=0):

        super().__init__(num_agents=num_agents,
                         max_iter=max_iter,
                         train_data=train_data,
                         train_label=train_label,
                         trans_func_shape=trans_func_shape,
                         save_conv_graph=save_conv_graph,
                         seed=seed)

        self.algo_name = 'MVO'
        self.agent_name = 'Universe'
        self.algo_params = {}


    def user_input(self):
        self.algo_params['Min'] = 0.2
        self.algo_params['Max'] = 1.0
        self.algo_params['p'] = 0.6


    def normalize(self, fitness):
        # normalize the fitness values
        fitness = np.asarray(fitness.reshape(1, -1), dtype=float)
        normalized_fitness = preprocessing.normalize(fitness, norm='l2', axis=1)
        normalized_fitness = np.reshape(normalized_fitness, -1)
        return normalized_fitness


    def roulette_wheel(self, fitness):
        # Perform roulette wheel selection
        maximum = sum([f for f in fitness])
        selection_probs = [f / maximum for f in fitness]
        return np.random.choice(len(fitness), p=selection_probs)


    def transfer_to_binary(self, i, j):
        if np.random.random() < self.trans_function(self.population[i, j]):
            self.population[i, j] = 1
        else:
            self.population[i, j] = 0

    def next(self):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(self.cur_iter + 1))
        print('================================================================================\n')

        Min = self.algo_params['Min']
        Max = self.algo_params['Max']
        p = self.algo_params['p']

        normalized_fitness = self.normalize(self.fitness)
        WEP = Min + (self.cur_iter * (Max - Min) / self.max_iter)  # Eq. (3.3)
        TDR = 1 - math.pow(self.cur_iter, (1 / p)) / math.pow(self.max_iter, (1 / p))  # Eq. (3.4)

        for i in range(self.num_agents):
            black_hole_idx = i
            for j in range(self.population.shape[1]):

                # Eq. (3.1)
                r1 = np.random.random()
                if r1 < normalized_fitness[i]:
                    white_hole_idx = self.roulette_wheel(-normalized_fitness)
                    self.population[black_hole_idx, j] = self.population[white_hole_idx, j]

                # Eq. (3.2)
                r2 = np.random.random()
                if r2 < WEP:
                    r3, r4 = np.random.random(2)
                    if r3 < 0.5:
                        self.population[i, j] = self.Leader_agent[j] + (TDR * r4)
                    else:
                        self.population[i, j] = self.Leader_agent[j] - (TDR * r4)

                # convert to binary using transfer function
                self.transfer_to_binary(i, j)

        # increment current iteration
        self.cur_iter += 1


if __name__ == '__main__':
    data = datasets.load_digits()
    algo = MVO(num_agents=20,
               max_iter=100,
               train_data=data.data,
               train_label=data.target,
               trans_func_shape='s')

    solution = algo.run()

