"""

Programmer: Rohit Kundu
Date of Development: 01/08/2021
This code has been developed according to the procedures mentioned in the following research article:
"Jain, M., Singh, V., & Rani, A. (2019). A novel nature-inspired algorithm for optimization: Squirrel search algorithm.
Swarm and evolutionary computation, 44, 148-175."

"""

import math

import numpy as np
from sklearn import datasets

from Py_FS.wrapper.nature_inspired._utilities_test import initialize as initialize_
from Py_FS.wrapper.nature_inspired._utilities_test import sort_agents as sort_agents_
from Py_FS.wrapper.nature_inspired.algorithm import Algorithm


class SSA(Algorithm):

    # Squirrel Search Algorithm
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of squirrels                                           #
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

        self.algo_name = 'SSA'
        self.agent_name = 'Squirrel'
        self.algo_params = {}

    def user_input(self):
        # initializing parameters
        pdp = self.algo_params['pdp'] = float(input('Predator Presence Probability: ') or 0.1)
        row = self.algo_params['row'] = float(input('Density of air: ') or 1.204)
        V = self.algo_params['V'] = float(input('Speed: ') or 5.25)
        S = self.algo_params['S'] = float(input('Surface area of body: ') or 0.0154)
        cd = self.algo_params['cd'] = float(input('Frictional Drag Coefficient: ') or 0.6)
        CL = self.algo_params['CL'] = float(input('Lift Coefficient: ') or 0.7)
        hg = self.algo_params['hg'] = float(input('Loss in height occurred after gliding: ') or 1)
        sf = self.algo_params['sf'] = float(input('Scaling Factor: ') or 18)
        Gc = self.algo_params['Gc'] = float(input('Gliding Constant: ') or 1.9)

    def initialize(self):
        super(SSA, self).initialize()
        self.squirrels_a = initialize_(num_agents=self.num_agents, num_features=self.num_features)
        self.squirrels_b = initialize_(num_agents=self.num_agents, num_features=self.num_features)
        self.squirrels_new, self.fitness = sort_agents_(self.squirrels_a, self.fitness)
        self.ind = np.where(self.fitness == max(self.fitness))[0]
        D1 = self.algo_params['D1'] = 1 / ((2 * self.algo_params['row'] * self.algo_params['V']) ** (2 * self.algo_params['S'] * self.algo_params['cd']))
        L = self.algo_params['L'] = 1 / ((2 * self.algo_params['row'] * self.algo_params['V']) ** (2 * self.algo_params['S'] * self.algo_params['CL']))
        tan_phi = self.algo_params['tan_phi'] = D1 / L
        dg = self.algo_params['dg'] = self.algo_params['hg'] / (tan_phi * self.algo_params['sf'])

    def check_squirrels(self, s):
        num_feat = s.shape[1]
        for i, squirrel in enumerate(s):
            if np.sum(squirrel) == 0:
                s[i] = np.random.randint(low=0, high=2, size=(num_feat,))
        return s

    def get_squirrel(self, agent, alpha=np.random.randint(-2, 3)):
        features = len(agent)
        lamb = np.random.uniform(low=-3, high=-1, size=(features))
        levy = np.zeros((features))
        get_test_value = 1 / (np.power((np.random.normal(0, 1)), 2))
        for j in range(features):
            levy[j] = np.power(get_test_value, lamb[j])  # Eq 2
        for j in range(features):
            agent[j] += (alpha * levy[j])  # Eq 1
        return agent

    def next(self):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(self.cur_iter + 1))
        print('================================================================================\n')

        pdp = self.algo_params['pdp']
        row = self.algo_params['row']
        V = self.algo_params['V']
        S = self.algo_params['S']
        cd = self.algo_params['cd']
        CL = self.algo_params['CL']
        hg = self.algo_params['hg']
        sf = self.algo_params['sf']
        Gc = self.algo_params['Gc']
        D1 = self.algo_params['D1']
        L = self.algo_params['L']
        tan_phi = self.algo_params['tan_phi']
        dg = self.algo_params['dg']

        for i in range(self.num_agents):
            R1 = np.random.random()
            if R1 >= pdp:
                self.squirrels_a[i, :] = np.round(
                    self.squirrels_a[i, :] + (dg * Gc * abs(self.squirrels_new[i, :] - self.squirrels_a[i, :])))
            else:
                self.squirrels_a[i, :] = initialize_(1, self.num_features)
            Fh = self.squirrels_a.copy()
            _, fit1 = sort_agents_(self.squirrels_a, self.fitness)

            R2 = np.random.random()
            if R2 >= pdp:
                self.squirrels_b[i, :] = np.round(
                    self.squirrels_b[i, :] + (dg * Gc * abs(self.squirrels_a[i, :] - self.squirrels_b[i, :])))
            else:
                self.squirrels_b[i, :] = initialize_(1, self.num_features)
            Fa = self.squirrels_b.copy()
            squirrels_new2, fit2 = sort_agents_(self.squirrels_b, self.fitness)

            R3 = np.random.random()
            if (R3 >= pdp):
                self.squirrels_b[i, :] = np.round(
                    self.squirrels_b[i, :] + (dg * Gc * abs(Fh[i, :] - self.squirrels_b[i, :])));
            else:
                self.squirrels_b[i, :] = initialize_(1, self.num_features)
            squirrels_new3, fit3 = sort_agents_(self.squirrels_b, self.fitness)

        self.population = (self.squirrels_a + self.squirrels_b) / 2

        Sc = np.sqrt(np.sum(np.abs(Fh - Fa)) ** 2);
        Smin = ((10 * math.exp(-6)) / (365)) ** (self.cur_iter / (self.max_iter / 2.5));
        if Sc < Smin:
            for j in range(self.population.shape[0]):
                levy_flight = np.random.uniform(low=-2, high=2, size=(self.num_features))
                levy_flight = self.get_squirrel(levy_flight)
                for k in range(self.num_features):
                    if self.trans_function(levy_flight[k]) > np.random.random():
                        self.population[j, k] = 1
                    else:
                        self.population[j, k] = 0
        else:
            for j in range(self.population.shape[0]):
                for k in range(self.population.shape[1]):
                    if self.trans_function(self.population[j, k]) > np.random.random():
                        self.population[j, k] = 1
                    else:
                        self.population[j, k] = 0

        self.population = self.check_squirrels(self.population)
        self.population, self.fitness = sort_agents_(self.population, self.fitness)

        # increment current iteration
        self.cur_iter += 1


if __name__ == '__main__':
    data = datasets.load_digits()
    algo = SSA(num_agents=20,
               max_iter=100,
               train_data=data.data,
               train_label=data.target,
               trans_func_shape='s')

    solution = algo.run()
