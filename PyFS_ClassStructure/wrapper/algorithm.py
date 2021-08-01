import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC as SVM
from sklearn.ensemble import RandomForestClassifier as RF

from PyFS_ClassStructure.wrapper.data import Solution, Data


class Algorithm():

    def __init__(self, num_agents, max_iter, features, labels, classifier):
        self.num_agents = num_agents
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.num_features = features.shape[1]
        self.max_iter = max_iter
        self.population = self.initialize()
        self.agents = self.population.copy()
        self.convergence_curve = {}
        self.data = self.split_data()
        self.agent_name = 'Agent'

        if classifier.upper() in ['KNN', 'SVM', 'RF']:
            self.classifier = eval(classifier.upper())
        else:
            self.classifier = None
            print(f'\n[Error!] We don\'t currently support {classifier.upper()} classifier...\n')
            exit(1)


    def initialize(self):
        # define min and max number of features
        min_features = int(0.3 * self.num_features)
        max_features = int(0.6 * self.num_features)

        # initialize the agents with zeros
        agents = np.zeros((self.num_agents, self.num_features))

        # select random features for each agent
        for agent_no in range(self.num_agents):
            # find random indices
            cur_count = np.random.randint(min_features, max_features)
            temp_vec = np.random.rand(1, self.num_features)
            temp_idx = np.argsort(temp_vec)[0][0:cur_count]

            # select the features with the ranom indices
            agents[agent_no][temp_idx] = 1

        return agents


    def split_data(self):
        # splitting data into training and validation
        val_size = float(input('Please enter validation set size [0-1]: '))
        if val_size <= 0 or val_size >= 1:
            val_size = 0.2 # standard train/val split
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=val_size, stratify=self.labels, random_state=0)

        data = Data()
        data.train_X = X_train.copy()
        data.val_X = X_test.copy()
        data.train_Y = y_train.copy()
        data.val_Y = y_test.copy()

        return data


    def sort_agents(self, obj):
        train_X, val_X, train_Y, val_Y = self.data.train_X, self.data.val_X, self.data.train_Y, self.data.val_Y



    def assign_fitness(self):
        pass


    def display_agents(self):
        pass


    def make_solution(self):
        soln = Solution()
        soln.num_features = self.num_features
        soln.num_agents = self.num_agents
        soln.max_iter = self.max_iter
        soln.convergence_curve = self.convergence_curve
        return soln


    def plot_convergence(self):
        # plot convergence curves
        num_iter = len(convergence_curve['fitness'])
        iters = np.arange(num_iter) + 1
        fig, axes = plt.subplots(1)
        fig.tight_layout(pad=5)
        fig.suptitle('Convergence Curves')

        axes.set_title('Convergence of Fitness over Iterations')
        axes.set_xlabel('Iteration')
        axes.set_ylabel('Avg. Fitness')
        axes.plot(iters, convergence_curve['fitness'])

        return fig, axes


if __name__ == '__main__':
    data = datasets.load_digits()
    alg = Algorithm(num_agents=10, max_iter=20, features=data.data, labels=data.target, classifier='KNN')
    print(alg.__dict__)