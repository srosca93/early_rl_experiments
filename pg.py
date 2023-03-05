import gym
import numpy as np
from pg_net import FullyConnectedNet
import matplotlib.pyplot as plt

class PolicyGradient():

    def __init__(self, env):
        self.env = gym.make(env)
        self.num_inputs = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.net = FullyConnectedNet([self.num_inputs, 24, self.num_actions], 0.000025)

        self.num_episodes = 5000
        self.gamma = 0.99

    def __del__(self):
        self.env.close()

    def get_action(self, observation):
        scores = self.net.forward_pass(observation.reshape(-1,4))
        flat_scores = np.squeeze(np.asarray(scores))
        exps = np.exp(flat_scores)
        probs = exps/np.sum(exps)
        action = np.random.choice(self.num_actions, p=probs)
        # print(probs)
        return action, probs

    def train(self):
        runs = []
        for ep in range(self.num_episodes):
            observation = self.env.reset()
            observations = []
            rewards = []
            grads = []
            total_reward = 0
            while True:
                observations.append(observation)
                action, probs = self.get_action(observation)
                # if ep > 9000:
                #     self.env.render()
                observation, reward, done, _ = self.env.step(action)

                if done == True and total_reward != 499:
                    reward = -100
                rewards.append(reward)
                total_reward += reward

                grad = np.zeros_like(probs)
                delta = np.zeros_like(probs)
                delta[action] = 1
                grad = delta - probs
                grads.append(grad)

                if done:
                    runs.append(total_reward)
                    break

            for i in range(len(grads)):
                dis_reward = np.sum([r * (self.gamma ** t) for t,r in enumerate(rewards[i:])])
                grad = grads[i] * dis_reward
                self.net.forward_pass(observations[i].reshape(-1,4))
                self.net.backward_pass(grad.reshape(-1,2))

            print("EP: " + str(ep) + " Score: " + str(total_reward) + "         ",end="\r", flush=False)

        plt.plot(runs)
        plt.show()

agent = PolicyGradient('CartPole-v1')
agent.train()
del agent
