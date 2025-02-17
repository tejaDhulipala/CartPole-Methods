import gymnasium as gym
from Agent import Agent
import numpy as np

class QRecorder:
    def __init__(self, env: gym.Env, agent: Agent, discount=1, epsilon=0.3, epsilon_decay=0.99):
        self.env = env
        self.recording = []
        self.agent = agent
        self.discount_factor = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def record_one_episode(self, last_episode=True, custom_reward = False):
        """
        Records one episode

        Returns a tuple in the form:
        (array of [(state, action, reward, state')], total return)
        """
        running = True
        observation, info = self.env.reset()
        episode = []
        return_value = 0
        discount = 1
        while running:
            action = self.agent.action(observation, 0.0)
            t_step = [observation, action]
            observation, reward, terminated, truncated, info = self.env.step(action)
            if custom_reward:
                reward = 0.418 - abs(observation[2])
            t_step = t_step + [reward]
            t_step = tuple(t_step + [observation])
            episode.append(t_step)
            return_value += reward * discount
            discount *= self.discount_factor

            if terminated or truncated:
                running = False
        
        if last_episode:
            self.env.close()

        return (episode, return_value)

    def record_episodes(self, N: int, custom_reward=False):
        """
        records N episodes to self.recording
        """
        for i in range(N):
            self.recording.append(self.record_one_episode(last_episode=(i==N-1), custom_reward=custom_reward))
    
    def reset(self):
        self.__init__(self.env, self.agent, discount=self.discount_factor)
    
    def sort_episodes(self):
        """
        changes the recording array such that the episodes are arranged in decreasing order
        """
        rewards = np.array(list(zip(*self.recording))[1])
        order = np.argsort(rewards)
        new_recording = []
        for i in range(len(self.recording)):
            new_recording.append(self.recording[order[i]])
        self.recording = new_recording   
    
    def percentile_episodes(self, percentile, is_destructive=False):
        """
        percentile: The top "percentile" of episodes to be considered
        if destrucitive is true, self.recording gets changed. Otherwise, it doesn't
        Either way it returns the array of episode tuples 
        """
        start = int(len(self.recording) * (1 - percentile / 100))
        top_episodes = self.recording[start:]
        if is_destructive:
            self.recording = top_episodes
        return top_episodes

    def change_epsilon(self, ep_decay, min_ep):
        self.epsilon = max(self.epsilon * ep_decay, min_ep)