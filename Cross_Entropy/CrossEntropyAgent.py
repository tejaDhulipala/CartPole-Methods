from torch import nn, tensor
import torch
from numpy import ndarray
import numpy as np
import gymnasium as gym
from Agent import Agent
from Recording import Recorder
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from Cross_Entropy.CrossEntropyDataset import CrossEntropyDataset
import time

class CrossEntropyAgent(Agent):
    def __init__(self, observation_space_size: int, action_space_size: int, num_hidden=128):
        super().__init__(observation_space_size, action_space_size)
        self.policy_network = nn.Sequential(
            nn.Linear(observation_space_size, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, action_space_size), # To get the actual answer, you have to apply sigmoid
        )
        self.logit_adjustment = nn.Softmax(0)

    def action(self, observation: ndarray):
        policy = self.logit_adjustment(self.policy_network(tensor(observation)))
        action = torch.multinomial(policy, num_samples=1)
        return action.item()

    def learn_one_epoch(self, data: DataLoader, learning_rate: float, optimizer: torch.optim, loss_func: torch.nn):
        loss = 0
        for idx, (input, output) in enumerate(data):
            optimizer.zero_grad()

            y_est = self.policy_network(input)
            loss = loss_func(y_est, output)
            loss.backward()

            optimizer.step()

            if idx == 0:
                print(loss)
        return loss
        
    def learn_batches(self, recording: list, batches: int, epochs: int, learning_rate: float):
        dataset = CrossEntropyDataset(recording)
        dataloader = DataLoader(dataset, batches, True)
        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate, amsgrad=True)
        # optimizer = torch.optim.SGD(self.policy_network.parameters(), lr = learning_rate)
        loss_func = nn.CrossEntropyLoss()
        loss = tensor(1)
        i = 0
        while loss.item() > 0.28 and i < epochs:
            loss = self.learn_one_epoch(dataloader, learning_rate, optimizer, loss_func)
            i += 1
        print("DONE LEARNING in " + str(i) + " epochs")
            
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
    agent = CrossEntropyAgent(4, 2)
    # agent.policy_network.load_state_dict(torch.load("cross_entropy_network.pth"))
    env = gym.make("CartPole-v1")
    recorder = Recorder(env, agent)
    test_recorder = Recorder(env, agent)
    num_episodes = 16
    med_val = []
    for i in range(10):
        print("Policy Iteration Number:", i)
        test_recorder.reset()
        recorder.reset()
        t0 = time.time()
        recorder.record_episodes(num_episodes, custom_reward=True, epsode_limit=2000)
        recorder.sort_episodes()
        test_recorder.record_episodes(num_episodes)
        test_recorder.sort_episodes()
        print(time.time() - t0)
        median_return = test_recorder.recording[int(num_episodes / 2)][1]
        print("Median Return: ", median_return)
        med_val.append(median_return)
        if median_return > 1500:
            torch.save(agent.policy_network.state_dict(), "../cross_entropy_network.pth")
            break
        recorder.percentile_episodes(70, True)
        agent.learn_batches(recorder.recording, 16, 1, 0.001)
    plt.plot(list(range(i + 1)), med_val, c="black")
    plt.title("Avg. Returns vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Avg. Return")
    plt.show()
    
    # a = input("")
    # recorder.record_episodes(100)
    # recorder.sort_episodes()
    # display_recording(recorder.recording)