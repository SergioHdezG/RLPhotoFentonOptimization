import random
import time
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from environments.env_base import EnvInterface, ActionSpaceInterface

class action_space(ActionSpaceInterface):
    def __init__(self, n_cities):
        """
        Actions: Permutaciones posibles
        """
        self.action_space = []
        for i in range(1, n_cities-1):
            self.action_space.append([i, i+1])

        self.action_space.append(None)


        self.n = len(self.action_space)
        self.action_space = np.array(self.action_space)


class Cities:
    def __init__(self, n_cities):
        self.n_cities = n_cities
        self.city_dist = np.random.randint(101, size=(n_cities, n_cities))
        # self.city_path = random.sample(range(1, n_cities), n_cities-1)
        self.max_dist_aprox = self._max_distance()

    def reset(self):
        self.city_path = random.sample(range(self.n_cities), self.n_cities)
        # self.city_path = [i for i in range(1, self.n_cities)]
        # self.city_path.insert(0, self.orig_city)
        # self.city_path.append(self.orig_city)

    def permute(self, index):
        if index is not None:
            self.city_path[index[0]], self.city_path[index[1]] = self.city_path[index[1]], self.city_path[index[0]]

    def get_path_distance(self):
        # sum = 0
        # for i in range(len(self.city_path)-1):
        sum = np.sum([self.get_disctance([self.city_path[i], self.city_path[i+1]]) for i in range(len(self.city_path)-1)])
        return sum

    def get_disctance(self, cities):
        return self.city_dist[min(cities[0], cities[1])][max(cities[0], cities[1])]

    def _max_distance(self):
        tril_inf = np.tril_indices(self.n_cities)
        self.city_dist[tril_inf] = 0
        return np.sum([max(x) for x in self.city_dist])

    def get_obs(self):
        # return self.cities[self.triu_sup]
        a = [self.get_path_distance()/self.max_dist_aprox]
        b = np.array(self.city_path)/(len(self.city_path)-1)
        return np.concatenate((a, b))

class env(EnvInterface):
    """ Problema del viajante
    """
    def __init__(self):
        super().__init__()
        n_cities = 6
        self.iterations = 0
        self.max_epochs = n_cities * 4

        self.action_space = action_space(n_cities)
        self.cities = Cities(n_cities)
        self.observation_space = np.zeros(n_cities+1,)

        self._rew = 0
        self.old_dist = 0
        self.glob_min_dist = 1e300

    def reset(self):
        """
        :return: observation. numpy array of state shape
        """
        self.iterations = 0
        self.cities.reset()
        self.old_dist = 0
        return self.cities.get_obs()

    def _act(self, action):
        act = self.action_space.action_space[action]
        self.cities.permute(act)
        self.iterations += 1

        return self.action_space.action_space[action] is None

    def step(self, action):
        """
        :param action:
        :return:
        """
        done = self._act(action)
        state = self.cities.get_obs()
        reward = self._reward(action)
        if not done:
            done = self._done()

        if done:
            self.render()

        return state, reward, done, None

    def render(self):
        print('Route: ', self.cities.city_path, ' Distance: ', self.cities.get_path_distance())

    def close(self):
        pass

    def _reward(self, action):
        dist = self.cities.get_path_distance()
        if dist < self.old_dist:
            self.old_dist = dist
            reward = (10/100) - self.iterations*0.025
        elif dist > self.old_dist:
            self.old_dist = dist
            reward = -(10/100) - self.iterations*0.025
        else:
            self.old_dist = dist
            reward = 0.

        if self.action_space.action_space[action] is None:
            # If current dist is less than global min dist plus 5%
            if dist < self.glob_min_dist * 1.15:
                # If current dist is actually less than global min dist
                if dist < self.glob_min_dist:
                    self.glob_min_dist = dist
                reward = (70/100) + (self.max_epochs - self.iterations)*self.max_epochs*0.05/self.max_epochs
            else:
                reward = -(10/100) - ((self.max_epochs - self.iterations)*(self.max_epochs*0.05/self.max_epochs))


        self._rew = reward

        return reward


    def _done(self):
        return self.iterations > self.max_epochs
