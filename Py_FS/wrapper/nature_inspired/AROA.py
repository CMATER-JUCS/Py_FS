"""
Programmer: Soumitri Chattopadhyay
Date of Development: 11/07/2021
This code has been developed according to the procedures mentioned in the following research article:
"Hashim, F.A., Hussain, K., Houssein, E.H. et al. Archimedes Optimization Algorithm.
Applied Intelligence, 51, 1531–1551 (2021)"
"""

import numpy as np
from sklearn import datasets

from Py_FS.wrapper.nature_inspired.algorithm import Algorithm


class AROA(Algorithm):

    # Archimedes Optimization Algorithm
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

        self.algo_name = 'AROA'
        self.agent_name = 'Particle'
        self.algo_params = {}

    def user_input(self):
        # initializing parameters
        self.algo_params['C1'] = float(input('Control variable C1 [1,2]: ') or 2)
        self.algo_params['C2'] = float(input('Control variable C2 [2,4,6]: ') or 6)
        self.algo_params['C3'] = float(input('Control variable C3 [1,2]: ') or 2)
        self.algo_params['C4'] = float(input('Control variable C4 [0-1]: ') or 0.5)
        self.algo_params['upper'] = float(input('upper limit for normalization [0-1]: ') or 0.9)
        self.algo_params['lower'] = float(input('lower limit for normalization [0-1]: ') or 0.1)

    def initialize(self):
        super(AROA, self).initialize()
        # initializing agent attributes
        self.position = np.random.rand(self.num_agents, self.num_features)  # Eq. (4)
        self.volume = np.random.rand(self.num_agents, self.num_features)  # Eq. (5)
        self.density = np.random.rand(self.num_agents, self.num_features)  # Eq. (5)
        self.acceleration = np.random.rand(self.num_agents, self.num_features)  # Eq. (6)

        # initializing leader agent attributes
        self.Leader_position = np.zeros((1, self.num_features))
        self.Leader_volume = np.zeros((1, self.num_features))
        self.Leader_density = np.zeros((1, self.num_features))
        self.Leader_acceleration = np.zeros((1, self.num_features))

        # rank initial agents
        self.sort_agents_attr()

    def sort_agents_attr(self):
        # sort the agents according to fitness
        if self.num_agents == 1:
            self.fitness = self.obj_function(self.population, self.training_data)
        else:
            fitnesses = self.obj_function(self.population, self.training_data)
            idx = np.argsort(-fitnesses)
            self.population = self.population[idx].copy()
            self.fitness = fitnesses[idx].copy()
            self.position = self.position[idx].copy()
            self.density = self.density[idx].copy()
            self.volume = self.volume[idx].copy()
            self.acceleration = self.acceleration[idx].copy()

        self.Leader_agent = self.population[0].copy()
        self.Leader_fitness = self.fitness[0].copy()
        self.Leader_position = self.position[0].copy()
        self.Leader_volume = self.volume[0].copy()
        self.Leader_density = self.density[0].copy()
        self.Leader_acceleration = self.acceleration[0].copy()

    def exploration(self, i, j, Df):
        C1 = self.algo_params['C1']

        # update acceleration
        rand_vol, rand_density, rand_accn = np.random.random(3)
        self.acceleration[i][j] = (rand_density + rand_vol * rand_accn) / (self.density[i][j] * self.volume[i][j])
        # update position
        r1, rand_pos = np.random.random(2)
        # Eq. (13)
        self.position[i][j] = self.position[i][j] + C1 * r1 * Df * (rand_pos - self.position[i][j])

    def exploitation(self, i, j, Tf, Df):
        C2 = self.algo_params['C2']
        C3 = self.algo_params['C3']
        C4 = self.algo_params['C4']

        # update acceleration
        self.acceleration[i][j] = (self.Leader_density[j] + self.Leader_volume[j] * self.Leader_acceleration[j]) / (
                self.density[i][j] * self.volume[i][j])
        # update position
        r2, r3 = np.random.random(2)
        T_ = C3 * Tf
        P = 2 * r3 - C4
        # Eq. (15)
        F = 1 if P <= 0.5 else -1
        # Eq. (14)
        self.position[i][j] = self.position[i][j] + F * C2 * r2 * self.acceleration[i][j] * Df * (
                (T_ * self.Leader_position[j]) - self.position[i][j])

    def normalize_accn(self, i, j):
        upper = self.algo_params['upper']
        lower = self.algo_params['lower']

        # Normalize accelerations
        max_accn = np.amax(self.acceleration[i])
        min_accn = np.amin(self.acceleration[i])

        # Eq. (12)
        self.acceleration[i][j] = lower + (self.acceleration[i][j] - min_accn) / (max_accn - min_accn) * upper

    def transfer_to_binary(self, i, j):
        # lower acceleration => closer to equilibrium
        if self.trans_function(self.acceleration[i][j]) < np.random.random():
            self.population[i][j] = 1
        else:
            self.population[i][j] = 0

    def post_processing(self):
        super(AROA, self).post_processing()
        # update other leader attributes
        if self.fitness[0] > self.Leader_fitness:
            self.Leader_agent = self.population[0].copy()
            self.Leader_fitness = self.fitness[0].copy()
            self.Leader_position = self.position[0].copy()
            self.Leader_volume = self.volume[0].copy()
            self.Leader_density = self.density[0].copy()
            self.Leader_acceleration = self.acceleration[0].copy()

    def next(self):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(self.cur_iter + 1))
        print('================================================================================\n')

        # weight factors
        Tf = np.exp((self.cur_iter - self.max_iter) / self.max_iter)  # Eq. (8)
        Df = np.exp((self.max_iter - self.cur_iter) / self.max_iter) - (self.cur_iter / self.max_iter)  # Eq. (9)

        for i in range(self.num_agents):
            for j in range(self.num_features):
                # Eq. (7)
                r1, r2 = np.random.random(2)
                # update density
                self.density[i][j] = self.density[i][j] + r1 * (self.Leader_density[j] - self.density[i][j])
                # update volume
                self.volume[i][j] = self.volume[i][j] + r2 * (self.Leader_volume[j] - self.volume[i][j])

                if Tf <= 0.5:
                    # Exploration phase
                    self.exploration(i, j, Df)
                else:
                    # Exploitation phase
                    self.exploitation(i, j, Tf, Df)

                # normalize accelerations
                self.normalize_accn(i, j)

                # convert to binary using transfer function
                self.transfer_to_binary(i, j)

        # increment current iteration
        self.cur_iter += 1


if __name__ == '__main__':
    data = datasets.load_digits()
    algo = AROA(num_agents=20,
                max_iter=100,
                train_data=data.data,
                train_label=data.target,
                trans_func_shape='s')

    solution = algo.run()
