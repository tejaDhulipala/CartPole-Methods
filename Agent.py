import numpy as np
from numpy import ndarray

class Agent:
    def __init__(self, observation_space_size: int, action_space_size: int):
        self.observation_size = observation_space_size
        self.action_size = action_space_size
    
    def action(self, observation: ndarray):
        return int(np.random.choice(np.arange(self.action_size)))