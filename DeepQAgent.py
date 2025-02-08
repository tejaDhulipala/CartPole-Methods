from torch import nn, tensor
import torch
from numpy import ndarray
import numpy as np
import gymnasium as gym
from Agent import Agent
from QRecording import QRecorder
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from QLearningDataset import QLearningDataset
from copy import deepcopy

class QAgent(Agent):
    def __init__(self, observation_space_size: int, action_space_size: int, num_hidden=128):
        super().__init__(observation_space_size, action_space_size)
        self.q_network = nn.Sequential(
            nn.Linear(observation_space_size, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, action_space_size)
        ).to(torch.float32)
        self.q_target = deepcopy(self.q_network).to(torch.float32)

    def action(self, observation: ndarray, epsilon):
        assert epsilon >= 0 and epsilon <= 1
        """
        takes action
        epsilon is the probabillity that it does the random action
        """
        if torch.rand(1).item() <= epsilon:
            return np.random.choice(np.arange(self.action_size))
        else:
            return torch.argmax(self.q_network.forward(torch.tensor(observation))).item()


    def learn_one_epoch(self, dataloader: DataLoader, loss_func: torch.nn.MSELoss, optimizer: torch.optim):
        for batch_num, ((state, action), (reward, next_state)) in enumerate(dataloader):
            optimizer.zero_grad()

            targets = reward + torch.max(self.q_target.forward(next_state))
            targets = targets.to(torch.float32)
            prediction = self.q_network.forward(state)
            prediction = prediction[np.arange(prediction.size(0)), action]

            loss = loss_func(prediction, targets).to(torch.float32)
            loss.backward()

            optimizer.step()

            if batch_num == 0:
                print(loss)

        return loss
        
    def learn_epochs(self, dataloader: DataLoader, learning_rate, epochs=1):
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate, amsgrad=True)
        for epoch in range(epochs):
            self.learn_one_epoch(dataloader, loss_func, optimizer)
    
    def update_target_network(self):
        self.q_target = deepcopy(self.q_network)   
            
def display_recording(recording):
    pass

def is_float(x):
    if isinstance(x, (float, np.floating)):  # Python and NumPy floats
        return True
    elif torch.is_tensor(x) and torch.is_floating_point(x):  # PyTorch tensors
        return True
    return False

if __name__ == "__main__":
    # Initilization
    torch.set_default_dtype(torch.float32)

    # Hyperparameters
    exploration_steps = 50
    train_itrs = 1 # How many times it trains on each training example sequence
    episodes_per_exploration = 20
    learning_rate = 0.0025
    target_update = 400 # in episodes
    batch_size = 10000
    
    # Fundamental objects
    env = gym.make("CartPole-v1")
    agent = QAgent(4, 2)
    agent
    agent.q_network.load_state_dict(torch.load("q_network.pth"))
    agent.q_target.load_state_dict(torch.load("q_network.pth"))

    recorder = QRecorder(env, agent, epsilon=1)
    experiences = QLearningDataset(recorder)
    median_returns = []
    for step in range(exploration_steps):
        recorder.record_episodes(episodes_per_exploration)
        experiences.update(batch_size) # called to record new experiences
        s = min(1000, (step + 1) * episodes_per_exploration)
        indices = np.random.choice(len(experiences), size=s)
        train_data = DataLoader(Subset(experiences, indices), batch_size=s)
        agent.learn_epochs(train_data, learning_rate)
        
        if (step + 1) * episodes_per_exploration % target_update < episodes_per_exploration:
            print("TARGET UPDATED")
            agent.update_target_network()
        
        recorder.change_epsilon(0.9, 0.01)
        
        # Display inforomation
        median_return = experiences.get_last_exploration_returns()
        median_return = list(sorted(median_return))[int(len(median_return) / 2)]
        print("Median Return on episode: ", step * episodes_per_exploration, median_return)
        print(recorder.epsilon)
        median_returns.append(median_return)

        if median_return > 150:
            torch.save(agent.q_network.state_dict(), "q_network.pth")
    
    plt.plot(list(range(0, exploration_steps * episodes_per_exploration, episodes_per_exploration)), median_returns, c="black")
    plt.title("Avg. Returns vs Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Avg. Return")
    plt.show()
