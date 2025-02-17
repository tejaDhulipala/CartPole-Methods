from Recording import Recorder
import torch
from torch.nn import Sequential, ReLU, Linear, Softmax
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from Cross_Entropy.CrossEntropyDataset import CrossEntropyDataset
import time

class ReinforceAgent:
    def __init__(self, observation_space_size: int, action_space_size: int, num_hidden=128):
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.policy_network = Sequential(
            Linear(observation_space_size, num_hidden),
            ReLU(),
            Linear(num_hidden, action_space_size), 
            Softmax(dim=0)
        )
    
    def action(self, observation_space: np.ndarray):
        policy = self.policy_network(torch.tensor(observation_space))
        action = torch.multinomial(policy, num_samples=1)
        return action.item() # returns index
    
    def learn(self, recording: list, learning_rate):
        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate, amsgrad=True)
        returns = np.array(list(zip(*recording))[1])
        returns = (returns - np.mean(returns)) # Quantifies how much better than baseline it is performing
        for idx, trajectory in enumerate(recording):
            optimizer.zero_grad()
            states, actions, rewards = list(zip(*trajectory[0]))
            actions = torch.tensor(actions, dtype=int)
            ret = returns[idx] if returns[idx] > 0 else 0 
            # print(len(states))
            # print(self.policy_network(torch.tensor(states)), actions)
            loss = -sum(ret * torch.log(self.policy_network(torch.tensor(states))[torch.arange(len(states)),actions]))
            loss.backward()
            print(loss)
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10)
            optimizer.step()



def display_recording(recording):
    eps = list(zip(*recording))[1]
    ep1 = plt.scatter(eps, np.arange(1, len(eps) + 1) / len(eps), color='red')
    # plt.scatter(shortened, np.arange(1, len(shortened) + 1) / len(shortened), color='blue')
    plt.title('Scatter Plot')
    plt.xlabel('Value')
    plt.ylabel('CDF: red, PDF: blue')
    plt.show()
    # print(eps)  

if __name__ == "__main__":
    agent = ReinforceAgent(4, 2)
    agent.policy_network.load_state_dict(torch.load("reinforce.pth")) 
    env = gym.make("CartPole-v1")
    recorder = Recorder(env, agent)
    test_recorder = Recorder(env, agent)
    num_episodes = 100
    med_val = []
    lr = 0.001
    for i in range(num_episodes):
        print("Episode:", i)
        test_recorder.reset()
        recorder.reset()
        t0 = time.time()
        recorder.record_episodes(10, custom_reward=False, epsode_limit=2000)
        # test_recorder.record_episodes(3) # This calculates the return without the custom reward function
        print(time.time() - t0)
        median_return = recorder.recording[int(len(recorder.recording) / 2)][1]
        print("Median Return: ", median_return)
        med_val.append(median_return)
        if median_return >= 1000:
            torch.save(agent.policy_network.state_dict(), "reinforce.pth")
            break
        t = time.time()
        agent.learn(recorder.recording, lr)
        print(time.time() - t)
    plt.plot(list(range(i + 1)), med_val, c="black")
    plt.title("Avg. Returns vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Avg. Return")
    plt.show()
    

    