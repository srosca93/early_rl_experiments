import gym
import numpy as np
from enum import Enum
import random
import copy
import matplotlib.pyplot as plt
from net import FullyConnectedNet

class eGreedy(Enum):
    EXPLORE = 0
    EXPLOIT = 1

class DQN():

    def __init__(self, env):
        self.env = gym.make(env)
        self.env._max_episode_steps = 200
        self.num_inputs = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.net = FullyConnectedNet(self.num_inputs, 32, 32, self.num_actions)
        self.prev_net = FullyConnectedNet(self.num_inputs, 32, 32, self.num_actions)

        self.memory_size = 2048
        self.memory_pointer = 0
        self.experiences = []
        self.batch_size = 32

        self.num_episodes = 500
        self.weight_update_interval = 4
        self.min_samples = self.batch_size
        self.epsilon = 1.0

    def __del__(self):
        self.env.close()

    def sample_experience(self):
        batch = random.sample(self.experiences, self.batch_size)
        obs_train  = np.array([batch[i][0] for i in range(len(batch))])
        act_train  = np.array([batch[i][1] for i in range(len(batch))])
        rew_train  = np.array([batch[i][2] for i in range(len(batch))])
        next_train = np.array([batch[i][3] for i in range(len(batch))])
        done_train = np.array([batch[i][4] for i in range(len(batch))])
        return obs_train, act_train, rew_train, next_train, done_train


    def get_action(self, observation):
        values = self.net.forward_pass(observation)
        population = list(eGreedy)
        probs = [self.epsilon, 1-self.epsilon]
        decision = random.choices(population, probs)[0]
        if decision == eGreedy.EXPLOIT:
            action = np.argmax(values)
        else:
            action = random.randint(0,1)
        return action

    def train(self):
        runs = []
        observation = self.env.reset()
        num_samples = 0
        for ep in range(self.num_episodes):
            total_reward = 0
            self.epsilon = max(0.1,self.epsilon * 0.995)
            while True:
                action = self.get_action(observation)
                next_observation, reward, done, info = self.env.step(action)
                if done == True and total_reward != 199:
                    reward = -1
                if self.epsilon < 0.11:
                    self.env.render() 
                if len(self.experiences) < self.memory_size:
                    self.experiences.append((observation, action, reward, next_observation, done))
                else:
                    self.experiences[self.memory_pointer] = (observation, action, reward, next_observation, done)
                if self.memory_pointer < self.memory_size-1:
                    self.memory_pointer += 1
                else:
                    self.memory_pointer = 0
                
                num_samples += 1
                total_reward += reward

                if num_samples > self.min_samples:
                    obs_train, act_train, rew_train, next_train, done_train = self.sample_experience()
                    next_values = self.prev_net.forward_pass(next_train).max(axis=1)
                    targets = np.zeros((self.batch_size,self.num_actions))
                    targets[np.arange(len(act_train)),act_train] = rew_train + 0.9*np.squeeze(np.asarray(next_values))
                    loss = self.net.train(obs_train, act_train, targets)

                if ((num_samples % self.weight_update_interval) == 0) & (num_samples > self.min_samples):
                    self.prev_net = copy.deepcopy(self.net)

                if done:
                    runs.append(total_reward)
                    observation = self.env.reset()
                    if ep % 10 == 0 and num_samples > self.min_samples:
                        print("Episode",ep,"complete. Reward =", total_reward, "steps =",num_samples,"loss =",loss)
                    break
                else:
                    observation = copy.deepcopy(next_observation)

        plt.plot(runs)
        plt.show()

agent = DQN('CartPole-v1')
agent.train()
del agent
