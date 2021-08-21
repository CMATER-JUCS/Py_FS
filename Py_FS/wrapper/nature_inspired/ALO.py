"""
Programmer: Rishav Pramanik
Date of Development: 20/08/2021
Antlion Optimizer
"""
import numpy as np
from wrapper.nature_inspired.algorithm import Algorithm
from wrapper.nature_inspired._transfer_functions import get_trans_function
from sklearn import datasets

class ALO(Algorithm):
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
        self.trans_func_shape=trans_func_shape
        super().__init__( num_agents=num_agents,
                        max_iter=max_iter,
                        train_data=train_data,
                        train_label=train_label,
                        save_conv_graph=save_conv_graph,
                        seed=seed )

        self.algo_name = 'ALO'
        self.agent_name = 'Antlion'
        self.trans_function = trans_func_shape
        self.algo_params = {}
        self.ants = initialize(self.num_agents, self.train_data.shape[-1]) #initialise the ants

    
    def user_input(self):
        pass

    
    
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
                population[i,j]=(x_random_walk[count] + e_random_walk[count])/2
        return population, antlions
    
    
    
    
    
    def next(self):
        print('\n================================================================================')
        print('\n                          Iteration - {}'.format(self.cur_iter + 1))
        print('================================================================================\n')
            
        self.ants, self.population = self.update_ants(self.ants, self.population,self.cur_iter,self.max_iter,self.weight_acc)
        self.ants, self.population = self.combine(self.ants, self.population)
        for i in range(self.num_agents):
            for j in range(self.train_data.shape[-1]):
                trans_value=self.trans_function(self.population[i,j])
                if (np.random.random() < trans_value): 
                    self.population[i,j] = 1
                else:
                    self.population[i,j] = 0
        self.cur_iter+=1
        compute_fitness(self.weight_acc)
        
        
        

if __name__ == '__main__':
    data=datasets.load_digits()
    alo = ALO(num_agents=50,
               max_iter=100,
               train_data=data.data,
               train_label=data.target,
               trans_func_shape='s')

    solution = alo.run()
