import gym
import numpy as np
from pg_net import FullyConnectedNet
import matplotlib.pyplot as plt

class ActorCritic():

    def __init__(self, env):
        self.env = gym.make(env)
        self.env._max_episode_steps = 500
        self.num_inputs = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.actor = FullyConnectedNet([self.num_inputs, 24, self.num_actions], 0.001)
        self.critic = FullyConnectedNet([self.num_inputs, 24, self.num_actions], 0.005, 0)

        self.num_episodes = 1000
        self.gamma = 0.99

    def __del__(self):
        self.env.close()

    def get_action(self, observation):
        scores = self.actor.forward_pass(observation.reshape(-1,4))
        flat_scores = np.squeeze(np.asarray(scores))
        flat_scores -= np.max(flat_scores)
        exps = np.exp(flat_scores)
        probs = exps/np.sum(exps)
        action = np.random.choice(self.num_actions, p=probs)
        return action, probs

    def train(self):
        runs = []
        for ep in range(self.num_episodes):
            # print("================================")
            observation = self.env.reset()
            total_reward = 0
            action, probs = self.get_action(observation)
            while True:
                # if ep > 100:
                #     self.env.render()

                observation_next, reward, done, _ = self.env.step(action)
                action_next, probs_next = self.get_action(observation_next)

                total_reward += reward

                target = np.zeros((1,self.num_actions))

                q_values = self.critic.forward_pass(observation.reshape(-1,4))

                q_values_next = self.critic.forward_pass(observation_next.reshape(-1,4))

                # if done == True and total_reward != 499:
                #     reward = -100
                if np.abs(np.max(q_values)) > 1000:
                    asdf
                target[0,action] = reward + self.gamma*np.max(q_values_next) - q_values[0,action]

                self.critic.forward_pass(observation.reshape(-1,4))
                dw = self.critic.backward_pass(target)

                # print("-----------")
                # print(target)
                # print(probs)
                # print(q_values)
                # print(q_values_next)
                # print(observation.reshape(-1,4))
                # print(dw)

                grad = np.zeros_like(probs)
                delta = np.zeros_like(probs)
                delta[action] = 1
                grad = delta - probs

                grad = grad * q_values[0,action]
                self.actor.forward_pass(observation.reshape(-1,4))
                self.actor.backward_pass(grad.reshape(-1,2))

                observation = observation_next
                action = action_next
                probs = probs_next

                if done:
                    runs.append(total_reward)
                    break

            if ep % 10 == 0:
                print("Episode",ep,"complete. Reward =", total_reward)

        plt.plot(runs)
        plt.show()

agent = ActorCritic('CartPole-v1')
agent.train()
del agent
