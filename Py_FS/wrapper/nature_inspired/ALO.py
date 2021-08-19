"""
Programmer: Rishav Pramanik
Date of Development: 20/08/2021
Antlion Optimizer
"""

from abc import abstractmethod
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from _utilities_test import Solution, Data, compute_accuracy, compute_fitness, initialize, sort_agents, display, call_counter
from wrapper.nature_inspired._transfer_functions import get_trans_function

class ALO():
    def __init__(self,
                 num_agents,
                 max_iter,
                 train_data,
                 train_label,
                 test_data=None,
                 test_label=None,
                 val_size=20,
                 trans_func_shape='s',
                 seed=0,
                 save_conv_graph=True,
                 max_evals=np.float("inf")):

        # essential user-defined variables
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.training_data = None
        self.train_data = train_data
        self.train_label = train_label
        self.trans_func_shape=trans_func_shape
        # user-defined or default variables
        self.test_data = test_data
        self.test_label = test_label
        self.val_size = val_size
        self.weight_acc = None
        self.seed = seed
        self.save_conv_graph = save_conv_graph

        # algorithm internal variables
        self.population = None
        self.num_features = None
        self.fitness = None
        self.accuracy = None
        self.Leader_agent = None
        self.Leader_fitness = float("-inf")
        self.Leader_accuracy = float("-inf")
        self.history = []
        self.cur_iter = 0
        self.max_evals = max_evals
        self.start_time = None
        self.end_time = None
        self.exec_time = None
        self.solution = None
        self.algo_name = 'Antlion Optimiser'
        self.agent_name = 'Antlion'


    
    def user_input(self):
        pass


    
    def next(self):
        print('\n================================================================================')
        print('\n                          Iteration - {}'.format(self.cur_iter + 1))
        print('================================================================================\n')
        #curr_count = 0
        #num_agents=self.num_agents
        #num_features=self.train_data.shape[-1]
        #while (curr_count < self.iterations):
            
        self.popul, self.population = self.update_ants(self.popul, self.population,self.cur_iter,self.max_iter,self.weight_acc)
        self.popul, self.population = self.combine(self.popul, self.population)
            #value = np.copy(antlions[antlions[:,-1].argsort()][0,:])
        for i in range(self.num_agents):
            for j in range(self.train_data.shape[-1]):
                trans_value=self.trans_function(self.population[i,j])
                if (np.random.random() < trans_value): 
                    self.population[i,j] = 1
                else:
                    self.population[i,j] = 0
            #print("Iteration = ", curr_count, "Best Fitness: ",fit[0])
        self.cur_iter+=1
        compute_fitness(self.weight_acc)
        #self.population=antlions
        #print('Best Antlion: ',elite_fitness, 'Best Accuracy: ',acc)
    def random_walk(self,iterations):
        x_random_walk = [0]*(iterations + 1)
        x_random_walk[0] = 0
        for k in range(1, len( x_random_walk)):
            rand = np.random.random()
            if rand > 0.5:
                rand = 1
            else:
                rand = 0
            x_random_walk[k] = x_random_walk[k-1] + (2*rand - 1)
        return x_random_walk
    def roulette_wheel(self,fitness):
        # Perform roulette wheel selection
        maximum = np.sum(fitness)
        selection_probs = [f/maximum for f in fitness]
        return np.random.choice(len(fitness), p=selection_probs)
    # Function: Combine Ants
    def combine(self,population, antlions):
        combination = np.vstack([population, antlions])
        combination = combination[combination[:,-1].argsort()]
        for i in range(0, population.shape[0]):
            for j in range(0, population.shape[1]):
                antlions[i,j]   = combination[i,j]
                population[i,j] = combination[i + population.shape[0],j]
        return population, antlions

    # Function: Update Antlion
    def update_ants(self,population, antlions, count, iterations,weight):
        i_ratio       = 1
        minimum_c_i   = np.zeros((1, population.shape[1]))
        maximum_d_i   = np.zeros((1, population.shape[1]))
        minimum_c_e   = np.zeros((1, population.shape[1]))
        maximum_d_e   = np.zeros((1, population.shape[1]))
        elite_antlion = np.zeros((1, population.shape[1]))
        if  (count > 0.10*iterations):
            i_ratio = (1000)*(count/iterations)
        elif(count > 0.50*iterations):
            i_ratio = (10000)*(count/iterations)
        elif(count > 0.75*iterations):
            i_ratio = (100000)*(count/iterations)
        elif(count > 0.90*iterations):
            i_ratio = (1000000)*(count/iterations)
        elif(count > 0.95*iterations):
            i_ratio = (10000000)*(count/iterations)
        for i in range (0, len(population)):
            fitness=compute_fitness(weight_acc=weight)
            #print(i)
            #fitness = fitness_function(antlions)
            ant_lion = self.roulette_wheel(self.fitness)
            for j in range (0, population.shape[1] - 1):
                '''
                minimum_c_i[0,j]   = antlions[antlions[:,-1].argsort()][0,j]/i_ratio
                maximum_d_i[0,j]   = antlions[antlions[:,-1].argsort()][-1,j]/i_ratio
                elite_antlion[0,j] = antlions[antlions[:,-1].argsort()][0,j]
                minimum_c_e[0,j]   = antlions[antlions[:,-1].argsort()][0,j]/i_ratio
                maximum_d_e[0,j]   = antlions[antlions[:,-1].argsort()][-1,j]/i_ratio
                '''
                minimum_c_i[0,j]   = antlions[0,j]/i_ratio
                maximum_d_i[0,j]   = antlions[-1,j]/i_ratio
                elite_antlion[0,j] = antlions[0,j]
                minimum_c_e[0,j]   = antlions[0,j]/i_ratio
                maximum_d_e[0,j]   = antlions[-1,j]/i_ratio
                rand = np.random.random()
                if (rand < 0.5):
                    minimum_c_i[0,j] =   minimum_c_i[0,j] + antlions[ant_lion,j]
                    minimum_c_e[0,j] =   minimum_c_e[0,j] + elite_antlion[0,j]
                else:
                    minimum_c_i[0,j] = - minimum_c_i[0,j] + antlions[ant_lion,j]
                    minimum_c_e[0,j] = - minimum_c_e[0,j] + elite_antlion[0,j]

                rand = np.random.random()
                if (rand >= 0.5):
                    maximum_d_i[0,j] =   maximum_d_i[0,j] + antlions[ant_lion,j]
                    maximum_d_e[0,j] =   maximum_d_e[0,j] + elite_antlion[0,j]
                else:
                    maximum_d_i[0,j] = - maximum_d_i[0,j] + antlions[ant_lion,j]
                    maximum_d_e[0,j] = - maximum_d_e[0,j] + elite_antlion[0,j]
                x_random_walk = self.random_walk(iterations)
                e_random_walk = self.random_walk(iterations)
                min_x, max_x = min(x_random_walk), max(x_random_walk)
                x_random_walk[count] = (((x_random_walk[count] - min_x)*(maximum_d_i[0,j] - minimum_c_i[0,j]))/(max_x - min_x)) + minimum_c_i[0,j]
                min_e, max_e = min(e_random_walk), max(e_random_walk)
                e_random_walk[count] = (((e_random_walk[count] - min_e)*(maximum_d_e[0,j] - minimum_c_e[0,j]))/(max_e - min_e)) + minimum_c_e[0,j]
                #population[i,j] = np.clip((x_random_walk[count] + e_random_walk[count])/2, min_values[j], max_values[j])
                population[i,j]=(x_random_walk[count] + e_random_walk[count])/2
            #population[i,-1] = target_function(population[i,0:population.shape[1]-1])
        return population, antlions
    
    def initialize(self):
        # set the objective function
        self.val_size = float(input('Percentage of data for valdiation [0-100]: ') or 30)/100
        self.weight_acc = float(input('Weight for the classification accuracy [0-1]: ') or 0.9)
        self.obj_function = call_counter(compute_fitness(self.weight_acc))

        # start timer
        self.start_time = time.time()
        np.random.seed(self.seed)

        # data preparation
        self.training_data = Data()
        self.train_data, self.train_label = np.array(self.train_data), np.array(self.train_label)
        self.train_label = self.int_encoding(self.train_label)
        self.training_data.train_X, self.training_data.val_X, self.training_data.train_Y, self.training_data.val_Y = train_test_split(self.train_data, self.train_label, stratify=self.train_label, test_size=self.val_size)

        # create initial population
        self.num_features = self.train_data.shape[1]
        self.population = initialize(num_agents=self.num_agents, num_features=self.num_features)
        self.fitness = self.obj_function(self.population, self.training_data)
        self.population, self.fitness = sort_agents(agents=self.population, fitness=self.fitness)
        self.accuracy = compute_accuracy(agents=self.population, data=self.training_data)
        self.Leader_agent, self.Leader_fitness = self.population[0], self.fitness[0]

    def int_encoding(self, labels):
        # converts the labels to one-hot-encoded vectors
        labels_str = np.array([str(i) for i in labels])

        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels_str)

        # # binary encode
        # onehot_encoder = OneHotEncoder(sparse=False)
        # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        # onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        
        return integer_encoded
    def check_end(self):
        # checks if the algorithm has met the end criterion
        return (self.cur_iter >= self.max_iter) or (self.obj_function.cur_evals >= self.max_evals)


    def save_details(self):
        # save some details of every generation
        cur_obj = {
            'population': self.population,
            'fitness': self.fitness,
            'accurcay': self.accuracy,
        }
        self.history.append(cur_obj)


    def display(self):
        # display the current generation details
        display(agents=self.population, fitness=self.fitness, agent_name=self.agent_name)


    def plot(self):
        # plot the convergence graph
        fig = plt.figure(figsize=(10, 8))
        avg_fitness = []
        for cur in self.history:
            avg_fitness.append(np.mean(cur['fitness']))

        plt.plot(np.arange(len(avg_fitness)), avg_fitness)
        plt.xlabel('Number of Generations')
        plt.ylabel('Average Fitness')
        plt.title('Convergence Curve')

        plt.show()

        return fig

    
    def post_processing(self):
        # post processing steps
        self.fitness = self.obj_function(self.population, self.training_data)
        self.population, self.fitness = sort_agents(agents=self.population, fitness=self.fitness)
        self.accuracy = compute_accuracy(agents=self.population, data=self.training_data)
        
        if(self.fitness[0] > self.Leader_fitness):
            self.Leader_fitness = self.fitness[0]
            self.Leader_agent = self.population[0, :]
            self.Leader_accuracy = self.accuracy[0]


    def save_solution(self):
        # create a solution object
        self.solution = Solution()
        self.solution.num_agents = self.num_agents
        self.solution.max_iter = self.max_iter
        self.solution.num_features = self.train_data.shape[1]
        self.solution.obj_function = self.obj_function

        # update attributes of solution
        self.solution.best_agent = self.Leader_agent
        self.solution.best_fitness = self.Leader_fitness
        self.solution.best_accuracy = self.Leader_accuracy
        self.solution.final_population = self.population
        self.solution.final_fitness = self.fitness
        self.solution.final_accuracy = self.accuracy
        self.solution.execution_time = self.exec_time

    def run(self):
        # the main algorithm run
        print('\n************    Please enter the values of the following paramters or press newline for using default values    ************\n')
        self.user_input()   # take the user inputs
        self.initialize()   # initialize the algorithm
        print('\n*****************************************************    Thank You    ******************************************************\n')
        self.popul = initialize(self.num_agents, self.train_data.shape[-1])
        #antlions   = initialize(self.num_agents, self.train_data.shape[-1])
        
        self.trans_function = get_trans_function(self.trans_func_shape)
        while(not self.check_end()):    # while the end criterion is not met
            self.next()                     # do one step of the algorithm
            self.post_processing()          # do the post processing steps
            self.display()                  # display the details of 1 iteration
            self.save_details()             # save the details

        self.end_time = time.time()     
        self.exec_time = self.end_time - self.start_time

        if self.test_data:          # if there is a test data, test the final solution on that 
            self.test_label = self.int_encoding(self.test_label)
            temp_data = Data()
            temp_data.train_X = self.train_data
            temp_data.train_Y = self.train_label
            temp_data.val_X = self.test_data
            temp_data.val_Y = self.test_label

            self.Leader_fitness = compute_fitness(self.Leader_agent, temp_data)
            self.Leader_accuracy = compute_accuracy(self.Leader_agent, temp_data)

        self.save_solution()
        fig = self.plot()

        if(self.save_conv_graph):
            fig.savefig('convergence_curve_' + self.algo_name + '.jpg')

        print('\n------------- Leader Agent ---------------')
        print('Fitness: {}'.format(self.Leader_fitness))
        print('Number of Features: {}'.format(int(np.sum(self.Leader_agent))))
        print('Accuracy of the fittest agent: {}'.format(self.Leader_accuracy))
        print('----------------------------------------\n')

        return self.solution


if __name__ == '__main__':
    data=datasets.load_digits()
    alo = ALO(num_agents=50,
               max_iter=100,
               train_data=data.data,
               train_label=data.target,
               trans_func_shape='s')

    solution = alo.run()
