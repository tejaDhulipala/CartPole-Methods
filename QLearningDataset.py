import torch
from torch.utils.data import Dataset, DataLoader
from QRecording import QRecorder

class QLearningDataset(Dataset):
    def __init__(self, q_recording: QRecorder):
        super().__init__()
        self.replay_buffer = []
        self.episode_lengths = []
        self.q_recording = q_recording
        self.returns = []
        for episode in q_recording.recording:
            i = 0
            for action_tuple in episode[0]:
                self.replay_buffer.append(action_tuple)
                i += 1
            self.episode_lengths.append(i)
            self.returns.append(episode[1])
    
    def __len__(self):
        return len(self.replay_buffer)

    def __getitem__(self, index):
        return self.replay_buffer[index][:2], self.replay_buffer[index][2:]

    def get_last_exploration(self):
        return self.replay_buffer[-1:-self.episode_lengths[-1] - 1:-1] 
    
    def get_last_exploration_returns(self):
        return self.returns[-1:-self.episode_lengths[-1] - 1:-1] 
    
    def last_exploration_length(self):
        return self.episode_lengths[-1]
    
    def update(self, max_len):
        self.__init__(self.q_recording)
        if len(self) > max_len:
            for _ in range(len(self) - max_len):
                self.replay_buffer.pop(0)
        print(len(self))

                
