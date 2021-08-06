"""
Programmer: Soumitri Chattopadhyay
Date of Development: 11/07/2021
This code has been developed according to the procedures mentioned in the following research article:
"Laith A., Diabat A., Mirjalili S., Elaziz M.A., Gandomi A.H. The Arithmetic Optimization Algorithm.
Computer Methods in Applied Mechanics and Engineering, 376, 113609 (2021)"
"""

import math
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

from algorithm import Algorithm
from _utilities_test import compute_fitness, sort_agents, compute_accuracy


class AOA(Algorithm):

    # Arithmetic Optimization Algorithm
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
                 save_conv_graph=False,
                 trans_func_shape='s',
                 seed=0):

        super().__init__(num_agents=num_agents,
                         max_iter=max_iter,
                         train_data=train_data,
                         train_label=train_label,
                         save_conv_graph=save_conv_graph,
                         trans_func_shape=trans_func_shape,
                         seed=seed)

        self.algo_name = 'AOA'
        self.agent_name = 'Agent'
        self.algo_params = {}


    def user_input(self):
        # initializing parameters
        self.algo_params['Min'] = 0.1
        self.algo_params['Max'] = 0.9
        self.algo_params['EPS'] = 1e-6
        self.algo_params['alpha'] = 5
        self.algo_params['mu'] = 0.5


    def moa(self, Min, Max, max_iter, t):
        return Min + (Max - Min) * t / max_iter


    def mop(self, max_iter, t, alpha=5):
        return 1 - (math.pow(t, (1 / alpha)) / math.pow(max_iter, (1 / alpha)))


    def exploration(self, i, j, MoP):
        eps = self.algo_params['EPS']
        mu = self.algo_params['mu']

        # Eq. (3)
        r2 = np.random.random()
        if r2 >= 0.5:
            self.population[i, j] = self.Leader_agent[j] * (MoP + eps) * mu
        else:
            self.population[i, j] =self.Leader_agent[j] / (MoP + eps) * mu


    def exploitation(self, i, j, MoP):
        eps = self.algo_params['EPS']
        mu = self.algo_params['mu']

        # Eq. (5)
        r3 = np.random.random()
        if r3 >= 0.5:
            self.population[i, j] = self.Leader_agent[j] + MoP * mu
        else:
            self.population[i, j] = self.Leader_agent[j] - MoP * mu


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
        alpha = self.algo_params['alpha']

        # Eq. (2)
        MoA = self.moa(Min, Max, self.max_iter, self.cur_iter)

        # Eq. (4)
        MoP = self.mop(self.max_iter, self.cur_iter, alpha)

        for i in range(self.num_agents):
            for j in range(self.population.shape[1]):

                r1 = np.random.random()

                if r1 > MoA:
                    self.exploration(i, j, MoP)     # Exploration phase (M,D)
                else:
                    self.exploitation(i, j, MoP)    # Exploitation phase (A,S)

                # convert to binary using transfer function
                self.transfer_to_binary(i, j)

        # increment current iteration
        self.cur_iter += 1


if __name__ == '__main__':
    data = datasets.load_digits()
    algo = AOA(num_agents=20,
               max_iter=30,
               train_data=data.data,
               train_label=data.target,
               trans_func_shape='s')

    solution = algo.run()