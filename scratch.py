import gym
import numpy as np
from enum import Enum
import random
import copy
import matplotlib.pyplot as plt

class eGreedy(Enum):
    EXPLORE = 0
    EXPLOIT = 1

class CartPoleSarsaAgent():

    def __init__(self):
        self.weights = np.zeros((4,2))
        self.epsilon = 1
        self.env = gym.make('CartPole-v0')
        self.learning_rate = 0.01
        self.succeeded = 0
        self.a_select = np.zeros((2))

    def __del__(self):
        self.env.close()

    def get_value(self, observation, action):
        return observation.dot(self.weights)[action]

    def get_action(self, observation):
        values = observation.dot(self.weights)
        population = list(eGreedy)
        probs = [self.epsilon, 1-self.epsilon]
        decision = random.choices(population, probs)[0]
        if decision == eGreedy.EXPLOIT:
            action = np.argmax(values)
        else:
            action = random.randint(0,1)
        return action

    def run_episode(self, lam):
        self.epsilon *= 0.99
        observation = self.env.reset()
        action = self.get_action(observation)
        done = False
        t = 0
        total_reward = 0
        while done != True:
            self.a_select[action] += 1
            if self.succeeded > 500 and self.epsilon < 0.01:
                self.env.render()
            next_observation, reward, done, info = self.env.step(action)
            total_reward += reward
            if not done:
                next_action = self.get_action(next_observation)
                next_value = self.get_value(next_observation, next_action)
            else:
                next_value = 0
                #total_reward = 0
            current_value = self.get_value(observation, action)
            delta = total_reward + next_value - current_value
            delta2 = -self.get_value(observation, ~action)
            self.weights[:,action]  += self.learning_rate * delta * observation
            self.weights[:,~action] += self.learning_rate * delta2 * observation
            observation = copy.deepcopy(next_observation)
            action = copy.deepcopy(next_action)
            t += 1
        if t == 200:
            self.succeeded += 1
        return t

    def train(self, lam):
        t = []
        num_ep = 1000
        for i_episode in range(num_ep):
            t.append(self.run_episode(lam))
            # if i_episode % 100 == 0:
            #     print(self.weights)
        fig, ax = plt.subplots()
        ax.plot(range(num_ep),t)
        plt.show()

agent = CartPoleSarsaAgent()
agent.train(0)
del agent