import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CrossEntropyDataset(Dataset):
    def __init__(self, recording: list):
        """
        Creates a dataset that represents one sequence of episodes, or one recording
        Recording should be of the form: [(episode1), (episode2), ...]
        Each episode should be of the form: ([(s1, a1, r1), (s2 a2, r2), ...], return)
        """
        super().__init__()

        if len(recording) > 0:
            assert isinstance(recording[0], tuple) and isinstance(recording[0][0], list) and len(recording[0][0][0]) == 3 and isinstance(recording[0][1], (float, int, np.float32))

        l = 0
        for ep in recording:
            l += len(ep[0])
        self.l = l

        self.observations = []
        self.actions = []
        for ep in recording:
            for pair in ep[0]:
                self.observations.append(pair[0])
                self.actions.append(pair[1])
        self.observations = torch.tensor(self.observations)
        self.actions = torch.tensor(self.actions)
    
    def __len__(self):
        """
        returns number of (s, a, r) pairs
        """
        return self.l

    def __getitem__(self, index):
        return self.observations[index], self.actions[index]